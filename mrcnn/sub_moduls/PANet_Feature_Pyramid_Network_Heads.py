import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from .ROIAlign_Layer import PyramidROIAlign_AFN, PyramidROIAlign
from .Utility_Functions import BatchNorm
############################################################
#  PANet Feature Pyramid Network Heads
############################################################


def panet_fpn_classifier_graph(rois, feature_maps, image_meta,
                               pool_size, num_classes, train_bn=True,
                               fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x2 = PyramidROIAlign_AFN([pool_size, pool_size], 2,
                             name="roi_align_classifier_2")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x2 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                            name="mrcnn_class_conv1_2")(x2)
    x2 = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1_2')(x2, training=train_bn)
    x2 = KL.Activation('relu')(x2)
    # 3
    x3 = PyramidROIAlign_AFN([pool_size, pool_size], 3,
                             name="roi_align_classifier_3")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x3 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                            name="mrcnn_class_conv1_3")(x3)
    x3 = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1_3')(x3, training=train_bn)
    x3 = KL.Activation('relu')(x3)
    # 4
    x4 = PyramidROIAlign_AFN([pool_size, pool_size], 4,
                             name="roi_align_classifier_4")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x4 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                            name="mrcnn_class_conv1_4")(x4)
    x4 = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1_4')(x4, training=train_bn)
    x4 = KL.Activation('relu')(x4)
    # 5
    x5 = PyramidROIAlign_AFN([pool_size, pool_size], 5,
                             name="roi_align_classifier_5")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x5 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                            name="mrcnn_class_conv1_5")(x5)
    x5 = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1_5')(x5, training=train_bn)
    x5 = KL.Activation('relu')(x5)
    x = KL.Add(name="mrcnn_mask_add_2_3")([x2, x3])
    x = KL.Add(name="mrcnn_mask_add_2_4")([x, x4])
    x = KL.Add(name="mrcnn_mask_add_2_5")([x, x5])

    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def panet_build_fpn_mask_graph(rois, feature_maps, image_meta,
                               pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x1 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                            name="mrcnn_mask_conv4_fc")(x)
    x1 = KL.TimeDistributed(BatchNorm(),
                            name='mrcnn_mask_conv4bn')(x1, training=train_bn)
    x1 = KL.Activation('relu')(x1)

    x1 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                            name="mrcnn_mask_conv5_fc")(x1)
    x1 = KL.TimeDistributed(BatchNorm(),
                            name='mrcnn_mask_conv5bn')(x1, training=train_bn)
    x1 = KL.Activation('relu')(x1)

    # x1 = KL.TimeDistributed(KL.Dense(256*4*4,activation="sigmoid"),
    #                       name="mrcnn_mask_fc")(x1)
    x1 = KL.TimeDistributed(KL.Flatten())(x1)
    x1 = KL.TimeDistributed(KL.Dense(28 * 28 * num_classes), name='mrcnn_mask_fc_logits')(x1)

    x1 = KL.Activation("softmax", name="mrcnn_class_fc")(x1)

    s = K.int_shape(x1)
    x1 = KL.Reshape((s[1], 28, 28, num_classes), name="mrcnn_mask_fc_reshape")(x1)
    # x1 = KL.TimeDistributed(KL.Reshape((14,14)),name="mrcnn_mask_fc_reshape")(x1)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)

    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="softmax"),
                           name="mrcnn_mask")(x)
    x = KL.Add(name="mrcnn_mask_add")([x, x1])
    x = KL.Activation('tanh', name="mrcnn_masksoftmax")(x)

    return x
