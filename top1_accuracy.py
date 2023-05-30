# Guillermo Enguita Lahoz 801618

# This script takes as input four files, one containing the ground truth annotations for a video (or set of videos)
# and the other with the predicted intervals, both for verbs and nouns.

# For each ground truth interval, the matches will be taken as any intervals in the same video with a IoU higher than
# a set threshold. Then, those intervals will be ordered by their score and the accuracy will be calculated
# taking only into account only the highest rated one.

import argparse
import pandas


# Prints top1 accuracy for the action predictions for a set of videos
# It takes as input the paths of the files containing the ground truth and the predicted intervals
def top1_accuracy(ground_truth_file, predictions_file):
    # Load intervals from file
    ground_truth = pandas.read_csv(ground_truth_file)
    predictions = pandas.read_csv(predictions_file)

    # Group predictions by video
    predictions_grouped = predictions.groupby(['video-id'])

    correct_matches = 0

    # For each ground truth instance
    for i in range(ground_truth.shape[0]):

        # For each predicted instance in the same video
        video_predictions = predictions_grouped.get_group(ground_truth['video-id'][i]).reset_index()
        for j in range(video_predictions.shape[0]):
            # We work out the Intersection over Union of the intervals
            union = max(ground_truth['t-end'][i], video_predictions['t-end'][j]) - \
                    min(ground_truth['t-start'][i], video_predictions['t-start'][j])
            intersection = min(ground_truth['t-end'][i], video_predictions['t-end'][j]) - \
                           max(ground_truth['t-start'][i], video_predictions['t-start'][j])
            iou = intersection / union

            # If the IoU is higher than a set threshold
            # We have found the highest rated predicted interval that matches the ground truth
            if iou >= 0.2:
                # If the labels are the same, count it as a match
                if video_predictions['label'][j] == ground_truth['label'][i]:
                    correct_matches += 1
                break

    # Print accuracy
    print("Top 1 Accuracy:", correct_matches/ground_truth.shape[0])


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='top1_accuracy.py',
        description='This script takes as input four files, one containing the ground truth annotations for a video ('
                    'or set of videos) and the other with the predicted intervals, both for verbs and nouns. '
                    'For each ground truth interval, the matches will be taken as any intervals in the same video '
                    'with a IoU higher than a set threshold. Then, those intervals will be ordered by their score and '
                    'the accuracy will be calculated taking only into account only the highest rated one.'
    )

    # Define the arguments used by the program
    parser.add_argument('ground_truth_nouns', help="Ground truth intervals (nouns) file name")
    parser.add_argument('predictions_nouns', help="Predicted intervals (nouns) file name")
    parser.add_argument('ground_truth_verbs', help="Ground truth intervals (verbs) file name")
    parser.add_argument('predictions_verbs', help="Predicted intervals (verbs) file name")
    args = parser.parse_args()

    top1_accuracy(args.ground_truth_verbs, args.predictions_verbs)
    top1_accuracy(args.ground_truth_nouns, args.predictions_nouns)


