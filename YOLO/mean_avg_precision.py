import tensorflow as tf
import numpy as np
from collections import Counter

from iou import intersection_over_union


def mean_avg_precision(prediciton_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """Calculate mean average precision

    Args:
        prediciton_boxes (list): list of all the bounding boxes described as [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        true_boxes (list): list of all the correct bounding boxes, similar to the prediction boxes
        iou_threshold (float, optional): threshold where the predicted bounding boxes is correct. Defaults to 0.5.
        num_classes (int, optional): number of classes. Defaults to 20.

    Returns:
        float: mean average precision across all classes for a specific IoU threshold
    """
    epsilon = 1e-6
    average_precisions = []
    ground_truths = []

    for c in range(num_classes):
        detections = []

        for prediction in prediciton_boxes:
            if prediction[1] == c:
                detections.append(prediction)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        num_bounding_boxes = Counter([ground_truth[0]
                                      for ground_truth in ground_truths])

        for key, value in num_bounding_boxes.items():
            num_bounding_boxes[key] = np.zeros(value)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_bounding_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bounding_box for bounding_box in ground_truths if bounding_box[0] == detection[0]]

            num_ground_truths = len(ground_truth_img)
            best_iou = 0

            for idx, ground_truth in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    tf.constant(detection[3:], dtype=tf.float32), tf.constant(ground_truth[3:], dtype=tf.float32))

                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_idx = idx

            if best_iou > iou_threshold:
                if num_bounding_boxes[detection[0]][best_ground_truth_idx] == 0:
                    TP[detection_idx] = 1
                    num_bounding_boxes[detection[0]][best_ground_truth_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumulative_sum = tf.cumsum(TP, axis=0)
        FP_cumulative_sum = tf.cumsum(FP, axis=0)
        recalls = TP_cumulative_sum / (total_true_bounding_boxes + epsilon)
        precisions = tf.divide(
            TP_cumulative_sum, (TP_cumulative_sum + FP_cumulative_sum + epsilon))
        precisions = tf.concat(
            (tf.constant([1], dtype=tf.float64), precisions), axis=0)
        recalls = tf.concat(
            (tf.constant([0], dtype=tf.float64), recalls), axis=0)
        average_precisions.append(np.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
