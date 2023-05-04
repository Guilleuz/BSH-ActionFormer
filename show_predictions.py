# Guillermo Enguita Lahoz, 801618
# This script takes as input two csv files, containing the ground truth actions and the predicted labels obtained
# from an action detection model, and for each 5 (or parameter) actions in the ground truth, it will print a graph
# containing the ground truth action intervals and the predicted intervals.

# The format for those files must be the following
# ,video-id,t-start,t-end,label
# <clip_num>,<video_num>,<start time in seconds>,<finish time in seconds>,<action label>
import argparse
import csv
import sys
import matplotlib.pyplot as plt
import mpld3
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


# Plots the ground truth and the predicted intervals in a video given its id
def plot_intervals(ground_truth_videos, prediction_videos, video_id, args):
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

    # Action colors, if max label is higher than the number of colors, there will be repeated colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

    max_colors = min(len(colors), max_label)

    # Get the colors for the intervals
    colors_ground = [colors[int(i % max_colors)] for i in labels_ground]
    colors_pred = [colors[int(i % max_colors)] for i in labels_pred]

    # Plot the intervals
    fig, ax = plt.subplots()
    ax.broken_barh(intervals, (0, 1), facecolors=colors_ground)
    ax.broken_barh(intervals_pred, (1, 1), facecolors=colors_pred)
    ax.set_ylim(0, 10)
    ax.set_xlim(minx, maxx)
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Ground-Truth', 'Predicted'])
    plt.show()

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

        # Update minx, maxx and max_label if neeeded
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

    # Load action intervals from each file onto memory, separated by video
    ground_truth_videos = load_intervals(ground_truth_file)
    prediction_videos = load_intervals(predictions_file, True, args.threshold)

    print("Loaded videos:")
    for video in ground_truth_videos:
        print(video)

    if "web" in args:
        plot_intervals(ground_truth_videos, prediction_videos, args.video_id, args)
    else:
        for video in ground_truth_videos:
            plot_intervals(ground_truth_videos, prediction_videos, video, args)


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
    parser.add_argument('--ground_truth', help="Ground truth intervals file name")
    parser.add_argument('--predictions', help="Predicted intervals file name")
    parser.add_argument("--threshold", type=float, default="0", help="If the predicted interval's score is lower"
                                                                     "than the threshold, it will be ignored.")
    parser.add_argument('--web', default=argparse.SUPPRESS, nargs='?', help="Use only when calling from streamlit,"
                                                                            "provides an interactive graph")
    parser.add_argument('--video_id', help="Name of the video to plot, only for streamlit")

    args = parser.parse_args()

    show_predictions(args.ground_truth, args.predictions, args)
