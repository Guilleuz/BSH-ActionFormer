# Guillermo Enguita Lahoz 801618
# This script takes as input the 'videos-cocinando-raw' dataset, as well as the 'BSH_video_annotations.json' file
# created by Miguel. That file is required as right now it is the only one that explicits the relation between the
# video file names and the video ids used on the annotation files.

# The script processes each one of the videos, and stores information about them in a json file.
# The script also takes as input a number from 0 to 100, indicating the chance of a video to be in the training set.
# Note: not every video is annotated


# The structure for the json output will be the following:
# [
#   {
#       "id": "video_id"          // video_id following the format: P01_<video number>
#       "name": "video_name.mp4", // The name used for the mp4 file
#       "framerate": "25",        // Framerate in fps
#       "duration": "201.03",     // Duration in seconds.hundreths
#       "height": "250",         // Video height in pixels
#       "width": "250",          // Video width in pixels
#       "subset": "training|validation", // Field indicating whether the video belongs to the training or validation set
#   },
#  {...},
#  ...
# ]

# Execute with:
# python process-video-info.py <path to video folder> <path to BSH_video_annotations.json>
#       <output_file> <training_set_chance>

import sys
import json
import cv2
import random


# Processes video information from video_folder, video ids are specified in video_annotations
# The training_set_chance indicates how likely a video is to be classified as a training set video
def process_videos(video_folder, video_annotations_, output_file_name, training_set_chance: int = 80):
    # Load the video annotations json file
    video_annotations_file = open(video_annotations_)
    video_annotations_json = json.load(video_annotations_file)
    videos = video_annotations_json['file']

    # Video data list
    video_list = []

    # Process each video
    for id in videos:
        # Extract information from the annotations file
        video_id = 'P01_' + id
        video_name = videos[id]['src']
        video_name = video_name[1:]
        video_path = video_folder + '/' + video_name

        # Open the video to get its framerate, duration and resolution
        # Requires opencv
        video = cv2.VideoCapture(video_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_seconds = total_frames / fps

        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Randomly choose if the video will be in the training or validation subset
        subset = 'training'
        if random.randint(0, 101) > int(training_set_chance):
            subset = 'validation'

        # Create a dictionary for the video and append it to the video list
        video_dict = {
            "id": video_id,
            "name": video_name,
            "framerate": fps,
            "duration": total_seconds,
            "height": height,
            "width": width,
            "subset": subset
        }
        video_list.append(video_dict)

    # Output all the video data to a json file
    output_file = open(output_file_name, 'w+')
    json.dump(video_list, output_file)


if __name__ == "__main__":
    process_videos(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

