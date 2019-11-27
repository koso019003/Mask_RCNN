def output_result_format(image_name, boxes, class_ids_, scores, output_file):
    # Number of instances
    num_ob = boxes.shape[0]
    if not num_ob:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids_.shape[0] == scores.shape[0]
    for ob in range(num_ob):
        y_min, x_min, y_max, x_max = boxes[ob]
        confidence = scores[ob]
        class_id = class_ids_[ob]

        output_result = "{} {} {} {} {} {} {}\n".format(
            image_name,
            x_min, y_min, x_max, y_max,
            class_id - 1,
            confidence
        )
        output_file.write(output_result)
