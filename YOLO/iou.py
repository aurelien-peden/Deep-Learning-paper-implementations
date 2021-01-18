import tensorflow as tf


def intersection_over_union(boxes_predictions, boxes_labels):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_predictions (tensor): Predictions of the bounding boxes with the following shape: (batch_size, 4)
        box_labels (tensor): Labels of the bounding boxes with the following shape: (batch_size, 4)

    Returns:
        tensor: Intersection over Union for all the bounding boxes
    """

    # The bounding boxes coordinates consists of the x, y
    # coordinates that represent the center of the box relative
    # to the whole image, and the width and height relative to the whole image
    boxes_predictions_x = boxes_predictions[..., 0:1]
    boxes_predictions_y = boxes_predictions[..., 1:2]
    boxes_predictions_w = boxes_predictions[..., 2:3]
    boxes_predictions_h = boxes_predictions[..., 3:4]
    boxes_predictions_x1 = boxes_predictions_x - boxes_predictions_w / 2
    boxes_predictions_y1 = boxes_predictions_y - boxes_predictions_h / 2
    boxes_predictions_x2 = boxes_predictions_x + boxes_predictions_w / 2
    boxes_predictions_y2 = boxes_predictions_y + boxes_predictions_h / 2

    boxes_labels_x = boxes_labels[..., 0:1]
    boxes_labels_y = boxes_labels[..., 1:2]
    boxes_labels_w = boxes_labels[..., 2:3]
    boxes_labels_h = boxes_labels[..., 3:4]
    boxes_labels_x1 = boxes_labels_x - boxes_labels_w / 2
    boxes_labels_y1 = boxes_labels_y - boxes_labels_h / 2
    boxes_labels_x2 = boxes_labels_x + boxes_labels_w / 2
    boxes_labels_y2 = boxes_labels_y + boxes_labels_h / 2

    # We use slicing so the shape is like (batch_size, 1)
    x1_intersection = tf.maximum(boxes_predictions_x1, boxes_labels_x1)
    y1_intersection = tf.maximum(boxes_predictions_y1, boxes_labels_y1)
    x2_intersection = tf.minimum(boxes_predictions_x2, boxes_labels_x2)
    y2_intersection = tf.minimum(boxes_predictions_y2, boxes_labels_y2)

    intersection_area = tf.clip_by_value(
        (x2_intersection - x1_intersection), 0, 1) * tf.clip_by_value(
            (y2_intersection - y1_intersection), 0, 1)

    boxes_predictions_area = (boxes_predictions_x2 - boxes_predictions_x1) * \
        (boxes_predictions_y2 - boxes_predictions_y1)
    boxes_labels_area = (boxes_labels_x2 - boxes_labels_x1) * \
        (boxes_labels_y2 - boxes_labels_y1)

    iou = intersection_area / (boxes_predictions_area + boxes_labels_area
                               - intersection_area + 1e-9)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou
