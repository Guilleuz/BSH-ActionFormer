# Guillermo Enguita Lahoz 801618

# This script takes as input the intervals predicted for a video (or videos),
# annotated with their start and end times, as well as their matching labels. It also requires an input file with all
# the ground truth intervals on the video(s). Furthermore, a csv file with the class name for each of the label
# classes is needed, as well as two threshold values. The first one, the Intersection over Union threshold,
# will limit the predicted intervals labeled as matches to those with a higher IoU value with the ground truth. The
# second one, the score threshold will determine the minimum confidence score for a predicted interval to be loaded.

# The script will generate a confusion matrix, showing what classes the original intervals have been given by the model.
# It can be called with:
# python confusion_matrix.py <ground truth file> <predictions file> <label names file> <IoU threshold> <score threshold>

import argparse
import csv

import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sn
from pandas import *


# Plots a confusion matrix for the given ground truth and predicted intervals
# Only intervals with a higher IoU to the ground truth than iou_threshold will be considered as matches
def get_confusion_matrix(ground_truth, predictions, label_names_file, iou_threshold, ignore_non_gt, title):
    # Get the unique labels in both ground truth and predictions
    unique_labels = pandas.unique(ground_truth['label'].tolist() + predictions['label'].tolist())
    unique_labels_gt = pandas.unique(ground_truth['label'].tolist())

    # Give them an id from 0 to range-1
    label_ids = {unique_labels[i]: i for i in range(unique_labels.shape[0])}
    confusion_matrix = np.zeros((len(unique_labels_gt), len(unique_labels)), dtype=int)

    # For each interval in the ground truth
    for i in range(ground_truth.shape[0]):
        gt_label = ground_truth['label'][i]

        # We calculate the IoU for every interval (from the same video) in the predictions
        for j in range(predictions.shape[0]):
            # If the predictions is not from the same video as the ground truth, we skip it

            if predictions['video-id'][j] != ground_truth['video-id'][i]:
                continue

            # We work out the Intersection over Union of the intervals
            union = max(ground_truth['t-end'][i], predictions['t-end'][j]) - \
                    min(ground_truth['t-start'][i], predictions['t-start'][j])
            intersection = min(ground_truth['t-end'][i], predictions['t-end'][j]) - \
                           max(ground_truth['t-start'][i], predictions['t-start'][j])
            iou = intersection / union

            # If the intersection is empty, we skip this interval
            if intersection <= 0.0:
                continue
            # Else, if the IoU is higher than a threshold
            elif iou > iou_threshold:
                # We count the predicted intervals label
                pred_label = predictions['label'][j]
                confusion_matrix[label_ids[gt_label], label_ids[pred_label]] += 1

    # Search for any empty columns
    empty_columns = list()

    # For every column
    for i in range(len(unique_labels)):
        empty = True

        # Check each element
        for j in range(len(unique_labels_gt)):

            # If an element is not 0, the column won't be empty
            if confusion_matrix[j][i] != 0:
                empty = False
                break

        if empty:
            empty_columns.append(i)

    # Remove the existing empty columns, as long as it is not in the ground truth
    for i, column in enumerate(empty_columns):
        if unique_labels[column] not in unique_labels_gt:
            confusion_matrix = np.delete(confusion_matrix, column, 1)
            unique_labels = np.delete(unique_labels, column)

            # Update the indexes of the following empty columns
            for j in range(i, len(empty_columns)):
                empty_columns[j] -= 1

    # Remove predictions not in a ground-truth class (if the --ignoreNonGT flag is set)
    if ignore_non_gt:
        # Delete the last column until we are done
        for i in range(0, len(unique_labels) - len(unique_labels_gt)):
            confusion_matrix = np.delete(confusion_matrix, -1, 1)
            unique_labels = np.delete(unique_labels, -1)

    # Load label names
    label_names = open(label_names_file, newline='\n')
    reader = csv.reader(label_names, delimiter=',', quotechar='|')
    label_names_dict = {int(row[0]): row[1] for row in reader}
    unique_label_names = [label_names_dict[label] for label in unique_labels]

    # Plot the confusion matrix
    df_cm = DataFrame(confusion_matrix, index=[i for i in unique_label_names[0:len(unique_labels_gt)]],
                      columns=[i for i in unique_label_names])
    sn.set(font_scale=1.5)  # OK for nouns
    fig = plt.figure(figsize=(10, 7))  # 15, 10
    s = sn.heatmap(df_cm, annot=True, cmap="crest", fmt="g", cbar=False, robust=True, linewidth=0.01, linecolor='w')
    s.set(xlabel='Predicted Class', ylabel='Ground Truth Class')
    plt.title(title, fontweight='bold', fontsize='20')
    plt.tight_layout()
    plt.show()


# Processes and loads the input intervals, both predicted and from the ground truth
# Only intervals with a confidence score higher than score threshold will be loaded
def process_input(ground_truth_file, predictions_file, label_names_file, iou_threshold, score_threshold, group_by_vid
                  , ignore_non_gt):
    # Load ground truth intervals
    ground_truth = read_csv(ground_truth_file)

    # Load predicted intervals, only if their confidence score is higher than a threshold
    predictions = (read_csv(predictions_file))[lambda x: x['score'] > score_threshold]
    predictions = predictions.reset_index()

    # If the groupByVideo flag is set, a confusion matrix will be plotted for each video
    if group_by_vid:
        # Group the intervals loaded by video id
        ground_truth = ground_truth.groupby(['video-id'])
        predictions = predictions.groupby(['video-id'])
        video_names = list(ground_truth.indices.keys())

        # Plot a confusion matrix for each group
        for video in video_names:
            video_ground_truth = ground_truth.get_group(video).reset_index()
            video_predictions = predictions.get_group(video).reset_index()
            get_confusion_matrix(video_ground_truth, video_predictions, label_names_file, iou_threshold, ignore_non_gt,
                                 "Confusion Matrix, video " + video)
    else:
        # Else, plot a united confusion matrix
        get_confusion_matrix(ground_truth, predictions, label_names_file, iou_threshold, ignore_non_gt,
                             "Confusion Matrix")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='confusion_matrix.py',
        description='This script takes as input two files, one with the ground truth intervals for the actions '
                    'performed in a video (or series of videos), and the other with the predicted intervals. '
                    'The script will generate a confusion matrix, showing for each class in the ground truth '
                    'the number of times that its intervals have been predicted as the different classes.'
    )

    # Define the arguments needed
    parser.add_argument('GroundTruthFile', help="Ground truth intervals file name")
    parser.add_argument('PredictionsFile', help="Predicted intervals file name")
    parser.add_argument('LabelNames', help="CSV file that contains in each row a label and its matching name")
    parser.add_argument('IoUThreshold', type=float, help="A predicted interval will only be considered as "
                                                         "a match with a ground truth interval if their IoU is higher "
                                                         "than this threshold")
    parser.add_argument('ScoreThreshold', type=float, help="Only predicted intervals with a higher confidence score "
                                                           "then the threshold set will be taken into account")
    parser.add_argument('--groupByVideo', action='store_true', default=argparse.SUPPRESS,
                        help="If this flag is set, the confusion matrix will be obtained for each video individually.")
    parser.add_argument('--ignoreNonGT', action='store_true', default=argparse.SUPPRESS,
                        help="If this flag is set, any prediction with a class not present in the ground truth"
                             "whill be ignored")

    # Parse the arguments
    args = parser.parse_args()

    # Process the inputs and call the necessary plots
    process_input(args.GroundTruthFile, args.PredictionsFile, args.LabelNames, args.IoUThreshold,
                  args.ScoreThreshold, "groupByVideo" in args, "ignoreNonGT" in args)
