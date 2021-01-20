import tensorflow as tf
from iou import intersection_over_union


def non_max_supression(predicted_boxes, iou_threshold, prob_threshold):
    """Perform Non Max Supression over the given bounding boxes

    Args:
        predicted_boxes (list): list of all the predicted bounding boxes described as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold for when a predicted bounding box is correct
        prob_threshold (float): threshold to remove predicted bounding boxes with low probabilities

    Returns:
        list: bounding boxes after performing non max supression
    """

    assert type(predicted_boxes) == list

    # Remove the prediction boxes that have a low probability of detecting
    # an object
    predicted_boxes = [
        box for box in predicted_boxes if box[1] > prob_threshold]

    predicted_boxes = sorted(
        predicted_boxes, key=lambda x: x[1], reverse=True)

    bounding_boxes_nms = []
    while predicted_boxes:
        highest_prob_pred_box = predicted_boxes.pop(0)

        # Remove the bounding boxes that have the same class as the highest probability bounding box
        # and that have a high IoU with that box
        predicted_boxes = [
            box for box in predicted_boxes if box[0] != highest_prob_pred_box[0]
            or intersection_over_union(
                tf.constant(highest_prob_pred_box[2:], dtype=tf.float32),
                tf.constant(box[2:], dtype=tf.float32)
            ) < iou_threshold
        ]

        bounding_boxes_nms.append(highest_prob_pred_box)

    # Return the list of the filtred bounding boxes
    return bounding_boxes_nms
