# Guillermo Enguita Lahoz 801618

"""
    This script takes as input a video annotations json file (for both verbs and nouns), and outputs a summary of
    each one of them, which includes, total length, number of action instances, temporal action density and number of
    different classes (both verbs and nouns) in each one of them.

    Furthermore, it will output an average value for the whole dataset, with the total length, average length, total
    and average number of instances and average temporal action density.

    The script can be called using:
    python video_summary.py <annotations_filename>

    The annotation file can contain the verb annotations or the noun annotations
"""
import json
import sys


def video_summary(annotations_filename):
    # Open and load the annotations json file
    annotations_file = open(annotations_filename)
    annotations_json = json.load(annotations_file)
    annotations_db = annotations_json['database']

    # Total length and number of instances for the whole dataset
    total_length = 0
    total_instances = 0

    # Set to count the amount of different classes seen throughout the dataset
    seen_classes = set()

    # Process each video's annotations
    for i in annotations_db:
        # Set to count the amount of different classes seen only in the current video
        video_seen_classes = set()

        # Read duration, annotations and number of instances
        duration = annotations_db[i]['duration']
        annottation_list = annotations_db[i]['annotations']
        num_instances = len(annottation_list)

        # Add duration and number of instances to the total
        total_length += duration
        total_instances += num_instances

        # Add each annotation class to the seen sets if not already present
        for annotation in annottation_list:
            video_seen_classes.add(annotation['label'])
            seen_classes.add(annotation['label'])

        # Print video summary
        print("Video ", i, ", length ", duration, "(s), number of instances ", num_instances, ", temporal density ",
              round(num_instances / duration, 3), "(actions per second)", ", different classes ",
              len(video_seen_classes), sep="")

    # Print dataset summary
    num_videos = len(annotations_db)
    print("\nDataset Summary")
    print("Number of videos: ", num_videos, ", Average length: ", round(total_length / num_videos, 2),
          ", Average number of instances: ", round(total_instances / num_videos, 2), ", Number of different classes: ",
          len(seen_classes), sep="")


if __name__ == "__main__":
    video_summary(sys.argv[1])
