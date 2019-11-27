"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained weights
    python3 pascal.py train --dataset=/path/to/coco/ --model=voc

    # Train a new model starting from ImageNet weights
    python3 pascal.py train --dataset=/path/to/voc/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 pascal.py train --dataset=/path/to/voc/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 pascal.py train --dataset=/path/to/voc/ --model=last

    # Run evaluatoin on the last model you trained
    python3 pascal.py evaluate --dataset=/path/to/voc/ --model=last
"""

import json
import os
import sys
import xmltodict
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib
import matplotlib.pyplot as plt
from output_result import output_result_format
from PIL import Image

sys.path.append('../../../')
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import model as modellib
from Mask_RCNN.mrcnn import panetmodel as panetmodellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.voc_eval import voc_ap

# Agg backend runs without a display
matplotlib.use('Agg')

# Inference result directory
RESULTS_DIR = os.path.abspath("./inference/")

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
VOC_MODEL_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class VOCConfig(Config):
    """Configuration for training on PASCAL VOC.
    Derives from the base Config class and overrides values specific
    to the PASCAL VOC dataset.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 200

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # VOC has 20 classes

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448


############################################################
#  Dataset
############################################################


class VOCDataset(utils.Dataset):

    def __init__(self):
        super(VOCDataset, self).__init__()

        self.stop_color = [np.array([0, 0, 0]), np.array([224, 224, 192])]
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person",
                        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.items_count = {c: 0 for c in self.classes}
        self.num_images = 0
        self._image_ids = None

    def load_voc(self, dataset_dir, mode):  # , class_id_=None, class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, mini_val, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different data sets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        if dataset_dir:
            pass
        assert mode in ('train', 'val', 'test')

        for ith, cls in enumerate(self.classes):
            self.add_class("voc", ith + 1, cls)

        # Path
        root_path = '../../../Data/datalab-cup2/'
        if mode == 'test':
            data_path = root_path + './pascal_voc_testing_data.txt'
            image_path = root_path + './VOCdevkit_test/VOCdevkit_test/VOC2007/JPEGImages/'
            annotation_dir = None

        else:
            data_path = root_path + './pascal_voc_training_data.txt'
            image_path = root_path + './VOCdevkit_train/VOCdevkit_train/VOC2007/JPEGImages/'
            annotation_dir = root_path + './VOCdevkit_train/VOCdevkit_train/VOC2007/Annotations/'

        num_val_from_train_set = 100
        val_random_seed = 1234

        def build_val_data():
            import random
            with open(data_path, 'r') as fr:
                lines = fr.readlines()
            random.seed(val_random_seed)
            val_lines = random.sample(lines, num_val_from_train_set)

            val_path = data_path.replace('training', 'val{}'.format(num_val_from_train_set))
            with open(val_path, 'w') as fw:
                for line_ in val_lines:
                    fw.write(line_)

            return val_path

        if mode == 'val':
            data_path = build_val_data()

        input_file = open(data_path, 'r')

        for ith, line in enumerate(input_file):
            line = line.strip()
            ss = line.split(' ')
            image_name = ss[0].replace('.jpg', '')
            this_image_path = os.path.join(image_path, image_name + '.jpg')

            record = np.asarray([float(num) for num in ss[1:]]).reshape([-1, 5])
            record_loc = record[:, :-1].astype(np.int32)
            record_tgt = record[:, -1].astype(np.int32) + 1
            object_num = len(record_tgt)

            if mode != 'test':
                with open(os.path.join(annotation_dir, image_name + ".xml")) as f:
                    anno = xmltodict.parse(f.read())
                    objs = anno['annotation']['object']
                    if not isinstance(objs, list):
                        objs = [objs]
                    for o in objs:
                        self.items_count[o['name']] += 1
                width = anno['annotation']["size"]['width']
                height = anno['annotation']["size"]['height']
            else:
                im = Image.open(this_image_path)
                width, height = im.size
                objs = None

            self.add_image(
                source='voc',
                image_id=ith,
                path=this_image_path,
                objs=objs,
                width=width,
                height=height,
                mask_loc=record_loc,
                class_ids=record_tgt,
                object_num=object_num,
            )
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def load_mask(self, image_id_):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id_]
        instance_masks = []
        class_ids_ = image_info['class_ids']

        mask_loc = image_info['mask_loc']
        for ith in range(mask_loc.shape[0]):
            xmin, ymin, xmax, ymax = mask_loc[ith]

            mask = np.zeros([int(image_info['height']), int(image_info['width'])], dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 1

            instance_masks.append(mask)

        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        return mask, class_ids_

    @property
    def image_ids(self):
        return self._image_ids


############################################################
#  Inference
############################################################

def inference(inference_model, dataset, limit, tag=''):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    time_dir = "{:%Y%m%dT%H%M%S}_{}".format(datetime.datetime.now(), tag)
    time_dir = os.path.join(RESULTS_DIR, time_dir)
    os.makedirs(time_dir)

    if limit < 0:
        limit = dataset.num_images
        print("inference all: {} images".format(limit))

    output_file = open('./test_prediction_{}.txt'.format(tag), 'w')

    # Load over images
    progress_bar = tqdm(dataset.image_ids[:limit])
    for image_id_ in progress_bar:
        # Load image and run detection
        image_ = dataset.load_image(image_id_)
        # Detect objects
        r_ = inference_model.detect([image_], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id_]["id"]
        # Save image with masks
        if len(r_['class_ids']) > 0:
            progress_bar.set_description(
                '[*] {}th image has {} instance(s).'.format(image_id_, len(r_['class_ids'])))
            visualize.display_instances(
                image_, r_['rois'], r_['masks'], r_['class_ids'],
                dataset.class_names, r_['scores'],
                show_bbox=True, show_mask=False,
                title="Predictions")
            plt.savefig("{}/{}".format(time_dir, source_id))
            plt.close()

            image_name = os.path.basename(dataset.image_info[image_id_]["path"])
            output_result_format(image_name, r_['rois'], r_['class_ids'], r_['scores'],
                                 output_file)
        else:
            plt.imshow(image_)
            plt.savefig("{}/noinstance_{}".format(time_dir, source_id))
            progress_bar.set_description(
                '[*] {}th image have no instance.'.format(image_id_))
            plt.close()

    output_file.close()


def visualize_test(dataset):
    image_ids = np.random.choice(dataset.image_ids, 20)
    test_dir = MODEL_DIR + "/visual_test"
    time_dir = "/{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    os.makedirs(test_dir, exist_ok=True)
    test_dir = test_dir + time_dir
    os.makedirs(test_dir, exist_ok=True)

    for image_id_ in image_ids:
        image_ = dataset.load_image(image_id_)
        mask_, class_ids_ = dataset.load_mask(image_id_)
        visualize.display_top_masks(image_, mask_, class_ids_, dataset.class_names)
        plt.savefig(test_dir + "/{}.png".format(image_id_))
        plt.close()


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Pascal VOC.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or 'inference' on VOC")
    parser.add_argument('--dataset',
                        metavar="/path/to/coco/",
                        default="/home/Datasets/PASCALVOC/VOCdevkit/VOC2012",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        default="init_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--use_pa',
                        action='store_true',
                        default=False, )
    parser.add_argument('--tag',
                        default="",
                        help="tag for experiment")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = VOCConfig()
    else:
        class InferenceConfig(VOCConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()

    if args.use_pa:
        config.NAME += "_PANet"
    if args.tag:
        config.NAME += "_{}".format(args.tag)

    config.display()

    if args.command == "train":
        command_mode = "training"
    else:
        command_mode = "inference"

    # Create model
    if args.use_pa:
        model = panetmodellib.PAMaskRCNN(mode=command_mode, config=config,
                                         model_dir=MODEL_DIR)
    else:
        model = modellib.MaskRCNN(mode=command_mode, config=config,
                                  model_dir=MODEL_DIR)
    # Select weights file to load
    if args.model.lower() == "voc":
        model_path = VOC_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VOCDataset()
        dataset_train.load_voc(args.dataset, "train")
        dataset_train.prepare()
        print("Items count in training set")
        print(json.dumps(dataset_train.items_count, indent=4))

        dataset_val = VOCDataset()
        dataset_val.load_voc(args.dataset, "val")
        dataset_val.prepare()
        print("Items count in evaluate set")
        print(json.dumps(dataset_val.items_count, indent=4))

        visualize_test(dataset_train)

        # This training schedule is an example. Update to fit your needs.

        if args.use_pa:
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=1000,
                        layers='all')
        else:
            # Training - Stage 1
            # Adjust epochs and layers as needed
            print("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=400,
                        layers='heads')

            # Training - Stage 2
            # Finetune layers from ResNet stage 4 and up
            print("Training Resnet layer 4+")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=1000,
                        layers='4+')

            # Training - Stage 3
            # Finetune layers from ResNet stage 3 and up
            print("Training Resnet layer 3+")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 100,
                        epochs=2000,
                        layers='all')

    elif args.command == "evaluate":
        dataset_val = VOCDataset()
        dataset_val.load_voc(args.dataset, "val")
        dataset_val.prepare()
        print("Items count in evaluate set")
        print(json.dumps(dataset_val.items_count, indent=4))

        correct_list = []
        score_list = []
        overlap_list = []

        for image_id in tqdm(dataset_val.image_ids):
            # Load image and ground truth data
            image, image_meta, class_ids, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset_val, config, image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            # Compute right or wrong
            co, sc, ov = utils.compute_mask_right_wrong(
                gt_bbox[:, :4], gt_mask, class_ids,
                r['rois'], r["masks"], r['class_ids'],
                r["scores"], overlap_thresh=0.5)

            # Merge them together
            correct_list.append(co)
            score_list.append(sc)
            overlap_list.append(ov)

        right = np.concatenate(correct_list).ravel()
        wrong = np.logical_not(right)
        score_list = np.concatenate(score_list).ravel()
        overlap_list = np.concatenate(overlap_list).ravel()

        order = (-score_list).argsort()

        tp = np.cumsum(right[order].astype(int))
        fp = np.cumsum(wrong[order].astype(int))
        rec = tp / float(len(order))
        prec = tp / (tp + fp)

        ap = voc_ap(rec, prec)
        mean_iou = overlap_list.sum() / float(tp[-1])

        print('Accuracy:\t{}'.format(tp[-1] / len(tp)))
        # print('AP:\t{}'.format(ap))
        print('Mean IoU:\t{}'.format(mean_iou))

    elif args.command == "inference" or args.command == "test":
        # print("evaluate have not been implemented")
        # Validation dataset
        dataset_val = VOCDataset()
        dataset_val.load_voc(args.dataset, "test")
        dataset_val.prepare()
        print("Running voc inference on {} images.".format(100))
        inference(model, dataset_val, int(100), tag=args.tag)

    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))
