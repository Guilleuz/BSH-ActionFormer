# Guillermo Enguita Lahoz, 801618
# This script takes as input two csv files, containing the ground truth actions and the predicted labels obtained
# from an action detection model, and for each 5 (or parameter) actions in the ground truth, it will print a graph
# containing the ground truth action intervals and the predicted intervals.

# The format for those files must be the following
# ,video-id,t-start,t-end,label
# <clip_num>,<video_num>,<start time in seconds>,<finish time in seconds>,<action label>

# Also, it requires a file that indicates the relationship between the labels used by the model, and the actual labels,
# in addition to a file containing the names of the actions according to their labels

# It allows the option to output the generated graph as an interactive html page, which can be enabled by using the
# option --web and calling the program with: streamlit run show_predictions.py -- [ARGS]
# The rest of the arguments can be shown using --help

# Run example:
# python show_predictions.py --ground_truth <ground csv file> --predictions <preds csv file> --threshold <value>
#   --separated

import argparse
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpld3
import numpy as np
import streamlit.components.v1 as components


# Loads the action intervals contained in intervals_file into a dictionary
# The dictionary contains an entry for each processed video, the key will be the video's name, and the value
# a list containing each interval
# If the file contains predicted intervals, only those with a score higher than score_threshold will be loaded
def load_intervals(intervals_file, is_pred=False, score_threshold=0.1):
    intervals = open(intervals_file)
    intervals_csv = csv.reader(intervals, delimiter=',')

    action_intervals = dict()
    # For each row in the file
    for row in intervals_csv:
        # Read the video id
        video_id = row[1]

        if video_id == "video-id":  # First line in the file, skip
            continue

        # If this is the first interval we read from a video
        if video_id not in action_intervals:
            # We create an empty list
            action_intervals.update({video_id: list()})

        # Get start, end times, and label from the interval
        start = float(row[2])
        end = float(row[3])
        label = int(row[4])

        # If the file has the prediction intervals, and the score is over a threshold
        # Or the file has ground truth intervals
        if (is_pred and float(row[5]) > score_threshold) or not is_pred:
            # Add the interval to the video's list
            interval = {'start': start, 'end': end, 'label': label}
            lista = action_intervals[video_id]
            lista.append(interval)
            action_intervals.update({video_id: lista})

    return action_intervals


# Action colors, if max label is higher than the number of colors, there will be repeated colors
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan', 'yellow', 'violet', 'palegreen', 'sandybrown', 'magenta', 'purple',
          'cyan', 'olivedrab', 'black', 'peru', 'darkblue']


# Creates a legend, including all the label names matching their respective colors
def get_legend(unique_labels, color_by_label, label_names):
    legend_handles = list()
    for label in reversed(unique_labels):
        handle = mpatches.Patch(color=color_by_label[label], label=label_names[label])
        legend_handles.append(handle)

    return legend_handles


# Plots the ground truth and the predicted intervals in a video given its id
def plot_intervals(ground_truth_videos, prediction_videos, video_id, label_names, args):
    # Get ground truth and prediction intervals for the given video
    ground_truth_intervals = ground_truth_videos[video_id]
    prediction_intervals = prediction_videos[video_id]

    minx = sys.float_info.max
    maxx = 0
    max_label = 0

    # Get the intervals in the format required for the plot
    # Also gets the max and min x value, the max label and the label arrays for the intervals
    intervals, max_label, maxx, minx, labels_ground = extract_intervals(max_label, maxx, minx, ground_truth_intervals)
    intervals_pred, max_label, maxx, minx, labels_pred = extract_intervals(max_label, maxx, minx, prediction_intervals)

    # If the matches_only flag is set, the prediction intervals with labels that do not appear in the ground-truth
    # will be hidden from the graph
    if "matches_only" in args:
        # Get a list with the labels in the ground truth (one appearence by label)
        ground_truth_labels = np.unique(labels_ground)

        # Filter all the predictions that have a label that does not appear in the ground truth
        intervals_pred = [intervals_pred[i] for i in range(len(intervals_pred)) if labels_pred[i] in ground_truth_labels]
        labels_pred = [labels_pred[i] for i in range(len(labels_pred)) if labels_pred[i] in ground_truth_labels]

    # Get a list with all the different labels, only one appearance by label
    unique_labels = np.unique(labels_ground + labels_pred)

    # Get the colors for the intervals
    # If the number of unique labels is higher than the number of colors, they will start repeating
    color_by_label = {int(label): colors[np.where(unique_labels == label)[0][0] % len(colors)]
                      for label in unique_labels}
    colors_ground = [color_by_label[i] for i in labels_ground]
    colors_pred = [color_by_label[i] for i in labels_pred]

    # Plot the intervals
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # Plots predicted and ground truth intervals
    ax.broken_barh(intervals, (0, 1), facecolors=colors_ground)

    # If the flag separated is set plot a barh for each label in the predictions
    if "separated" in args:
        # Get each unique label in the predictions
        unique_pred_labels = np.unique(labels_pred)

        # Init bar height, ytick positions and labels
        height = 1
        yticks = [0.5]
        ytick_labels = ['Ground']

        # For each label in the predictions
        for label in unique_pred_labels:
            # Get the intervals with that label
            intervals_label = [intervals_pred[i] for i in range(len(intervals_pred)) if labels_pred[i] == label]

            # Plot a barh with only those intervals
            ax.broken_barh(intervals_label, (height, 1), color=color_by_label[label], alpha=0.5)
            ytick_labels.append(label_names[label])
            yticks.append(height + 0.5)
            height += 1

        # Set ticks and vertical limit
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim(0, yticks[-1] + 0.5)
    else:
        # If the flag is not set, all the predicted intervals will be plotted in the same barh
        # Not recommended if there is a lot of overlap between the actions
        ax.broken_barh(intervals_pred, (1, 1), facecolors=colors_pred)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Ground', 'Predicted'])
        ax.set_ylim(0, 2)

    # Set graph limits
    ax.set_xlim(minx, maxx)

    # Set labels
    ax.set_xlabel('Time (seconds)')

    # Set legend, if the hide_legend flag is not set
    if "hide_legend" not in args:
        legend_handles = get_legend(unique_labels, color_by_label, label_names)
        plt.legend(handles=legend_handles)

    # Set title
    plt.title(video_id)
    plt.tight_layout()
    plt.show()

    # Plot using streamlit if desired
    if "web" in args:
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)


# From the interval list, extracts each individual interval and transforms it to the values required for the plot
# Also gets the maximum end time and the minimum start time, and an array of labels corresponding to the intervals
def extract_intervals(max_label, maxx, minx, action_intervals):
    intervals = list()
    labels = list()

    # For each interval
    for interval in action_intervals:
        # Convert to the format required by plt
        intervals.append((interval['start'], interval['end'] - interval['start']))
        labels.append(interval['label'])

        # Update minx, maxx and max_label if needed
        if interval['start'] < minx:
            minx = interval['start']

        if interval['end'] > maxx:
            maxx = interval['end']

        if interval['label'] > max_label:
            max_label = interval['label']

    return intervals, max_label, maxx, minx, labels


# Plots the ground truth and predicted action intervals in a video
def show_predictions(ground_truth_file, predictions_file, args):
    plt.ion()

    # Set matplotlib font
    font = {'weight': 'bold',
            'size': 22}
    plt.rc('font', **font)

    # Load action intervals from each file onto memory, separated by video
    ground_truth_videos = load_intervals(ground_truth_file)
    prediction_videos = load_intervals(predictions_file, True, args.threshold)

    # Load label names
    label_names_file = open(args.label_names, newline='\n')
    reader = csv.reader(label_names_file, delimiter=',', quotechar='|')
    label_names = {int(row[0]): row[1] for row in reader}

    # Print the names of the loaded videos
    print("Loaded videos:")
    for video in ground_truth_videos:
        print(video)

    if "web" in args or "video_id" in args:
        # If we are using streamlit, plot only one graph
        plot_intervals(ground_truth_videos, prediction_videos, args.video_id, label_names, args)
    else:
        # Else, plot a graph for each video
        for video in ground_truth_videos:
            plot_intervals(ground_truth_videos, prediction_videos, video, label_names, args)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='show_predictions.py',
        description='This script takes as input two csv files, containing the ground truth actions and the predicted'
                    ' labels obtained from an action detection model, and for each 5 (or parameter) actions in the '
                    'ground truth, it will print a graph containing the ground truth action intervals and the '
                    'predicted intervals.'
    )

    # Define the arguments used by the program
    parser.add_argument('--ground_truth', required=True, help="Ground truth intervals file name")
    parser.add_argument('--predictions', required=True, help="Predicted intervals file name")
    parser.add_argument('--label_names', required=True,
                        help="CSV file that contains in each row a label and its corresponding name")
    parser.add_argument("--threshold", type=float, default="0", help="If the predicted interval's score is lower"
                                                                     "than the threshold, it will be ignored.")
    parser.add_argument('--matches_only', default=argparse.SUPPRESS, nargs='?',
                        help="If this flag is set, the graph will only show predicted intervals with an action label "
                             "that appears in the ground truth, removing the rest.")
    parser.add_argument('--separated', default=argparse.SUPPRESS, nargs='?',
                        help="If this flag is set, the graph will only show an individual horizontal bar for each label"
                             "in the prediction set. Not recommended if the number of predicted classes is high.")
    parser.add_argument('--hide_legend', default=argparse.SUPPRESS, nargs='?',
                        help="If this flag is set, the graph will not show the legend. Recommended if the number of "
                             "classes is very high.")
    parser.add_argument('--web', default=argparse.SUPPRESS, nargs='?', help="Use only when calling from streamlit,"
                                                                            "provides an interactive graph")
    parser.add_argument('--video_id', default=argparse.SUPPRESS, help="Name of the video to plot, only for streamlit")

    args = parser.parse_args()

    show_predictions(args.ground_truth, args.predictions, args)
