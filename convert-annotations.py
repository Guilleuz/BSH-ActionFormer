# Guillermo Enguita Lahoz 801618
# Script to convert action video annotations in MotionFormer's format to ActionFormer's

# MotionFormer requires a csv file, each row will represent an action annotation, in the following format:
# narration_id, participant_id,	video_id, narration_timestamp, start_timestamp, stop_timestamp, start_frame,
#       stop_frame, narration, verb, verb_class, noun, noun_class, all_nouns, all_noun_classes
# e.g:
# P01_4_0,P01,P01_4,00:00:04.30,00:00:04.30,00:00:08.80,260,525,stir vegetable,stir,10,vegetable,94,['vegetable'],[94]

# Annotations corresponding to the same video are in adjacent rows

# For ActionFormer we require two json files, one of them will contain the verbs, while the other will contain the nouns
# e.g:
# {version: "dataset-name", database:
#   {"video1_id": {"annotations": [{"label": "3", "label_id": "3", "segment": [0.14, 3.37]}, ...]},
#       "resolution": "512x512", "duration": "105.2", "subset": "training|validation"},
#   {"video2_id": ...},
#   ...}

# Label/Label_id match to the action/noun label in the Epic Kitchens Dataset
#   We can extract them from the verb_class/noun_class fields
# In the field segment, we have to specify the starting and ending times of the action in <seconds>:<hundreths>
#   We can extract those from the start_timestamp and stop_timestamp fields

# Usage: python convert-annotations.py <input annotations> <video info file> <output name>
# input annotations: path to the csv file with the action annotations used for MotionFormer
# video info file: path to the json file containing the video metadata (id, filename, resolution, duration, subset, ...)

import sys
import csv
import json


# Converts an HH:MM:SS:HH timestamp to SS:HH
def get_seconds(timestamp):
    splits = timestamp.split(':')
    hours = float(splits[0])
    minutes = float(splits[1])
    seconds = float(splits[2])
    return str(hours * 3600 + minutes * 60 + seconds)


# Loads each videos metadata (id, filename, resolution, duration, subset and framerate) from a json file
# Returns a dictionary with the video id as a key, its duration, resolution and subset as values
def load_video_info(input_path):
    # Load json video info file
    video_info = open(input_path)
    videos = json.load(video_info)

    video_dict = {}

    # Create a dictionary with the video id as key, duration, resolution and subset as values
    for v in range(len(videos)):
        video_id = videos[v]['id']
        video_duration = videos[v]['duration']
        video_resolution = str(videos[v]['width']) + 'x' + str(videos[v]['height'])
        video_subset = videos[v]['subset']

        video_dict[video_id] = {
            "duration": video_duration,
            "resolution": video_resolution,
            "subset": video_subset
        }

    return video_dict


def convert_annotations(input_file, output_file, video_info_file):
    # Load video information
    video_dict = load_video_info(video_info_file)

    # Output file names, one for noun annotations, one for verb annotations
    output_file_nouns = output_file + "_nouns.json"
    output_file_verbs = output_file + "_verbs.json"

    # Open the input csv annotations file
    file = open(input_file)
    csv_reader = csv.DictReader(file, delimiter=',')

    # Previous video's name
    prev_video = ""
    # List of noun/verb annotations in a video
    noun_annotations = []
    verb_annotations = []

    # List of noun/verb information from different videos
    noun_database = dict()
    verb_database = dict()

    # Create the vid_list.csv file
    vid_list_csv = open('vid_list.csv', 'w+')

    for row in csv_reader:
        # If we read an action corresponding to a new video
        if prev_video != row["video_id"]:
            # If the last video was not empty, we save its annotations
            if prev_video != "":
                # Output the videos' name to the vid_list.csv
                vid_list_csv.write(prev_video + '.mp4\n')

                # Create two dictionaries with the videos' annotations, its resolution, duration and subset
                video_dict_nouns = {
                    prev_video: {
                        "annotations": noun_annotations,
                        "resolution": video_dict[prev_video]["resolution"],
                        "duration": video_dict[prev_video]["duration"],
                        "subset": video_dict[prev_video]["subset"]
                    }
                }

                video_dict_verbs = {
                    prev_video: {
                        "annotations": verb_annotations,
                        "resolution": video_dict[prev_video]["resolution"],
                        "duration": video_dict[prev_video]["duration"],
                        "subset": video_dict[prev_video]["subset"]
                    }
                }

                # Add the dictionaries to the noun and verb lists
                noun_database.update(video_dict_nouns)
                verb_database.update(video_dict_verbs)

            # Get the new videos' name, create empty annotation lists
            prev_video = row["video_id"]
            noun_annotations = []
            verb_annotations = []

        # Save the annotation in the nouns and the verbs list
        noun_id = row["noun_class"]
        verb_id = row["verb_class"]
        start_time = row["start_timestamp"]
        stop_time = row["stop_timestamp"]

        # Change timestamps from HH:MM:SS.HH to SS.HH
        start_time = get_seconds(start_time)
        stop_time = get_seconds(stop_time)

        # Build a dictionary with the noun annotation
        noun_dict = {
            "label": noun_id,
            "label_id": noun_id,
            "segment": [start_time, stop_time]
        }
        # Append the dictionary to the videos' list
        noun_annotations.append(noun_dict)

        # Build a dictionary with the verb annotation
        verb_dict = {
            "label": verb_id,
            "label_id": verb_id,
            "segment": [start_time, stop_time]
        }
        # Append the dictionary to the videos' list
        verb_annotations.append(verb_dict)

    # Create and write the final json files for noun and verb annotations
    noun_database_complete = {"version": output_file + "_noun", "database": noun_database}
    verb_database_complete = {"version": output_file + "_verb", "database": verb_database}

    noun_output = open(output_file_nouns, 'w+')
    json.dump(noun_database_complete, noun_output)
    verb_output = open(output_file_verbs, 'w+')
    json.dump(verb_database_complete, verb_output)
    vid_list_csv.close()


if __name__ == "__main__":
    convert_annotations(sys.argv[1], sys.argv[2], sys.argv[3])
