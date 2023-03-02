# Guillermo Enguita Lahoz 801618
# This script takes as input the 'videos-cocinando-raw' dataset.
# The script processes each one of the videos, and stores information about them in a json file, furthermore
# it changes the videos' names to fit the ones specified in the annotations file.
# It also takes as input a number from 0 to 100, indicating the chance of a video to be in the training set.
# Finally, it creates the file vid_list.csv, which contains all the videos' names and is required for the
# SlowFast Feature Extractor.

# Note: not every video is annotated
# Note: the video id is related to the order of the videos in the folder, so the first one to appear will be
#       P01_1, the second one P01_2 and so on


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
# python process-video-info.py <path to video folder> <output_file> <training_set_chance>

import sys
import json
import cv2
import random
import os


# Processes video information from video_folder, video ids are specified in video_annotations
# The training_set_chance indicates how likely a video is to be classified as a training set video
def process_videos(video_folder, output_file_name, training_set_chance: int = 80):
    # Read all the video files' names from the input folder
    video_names = os.listdir(video_folder)

    # Video data list
    video_list = []

    # vid_list.csv file, required by the feature extractor
    # It contains the relative path to all the videos
    vid_list_csv = open(video_folder + '/vid_list.csv', 'w+')

    # Process each video
    for video_number in range(1, len(video_names) + 1):
        # Extract information from the annotations file
        video_id = 'P01_' + str(video_number)
        video_name = video_names[video_number - 1]
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
        video.release()

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

        # Change the videos' name to the one we will use for the feature extractor
        os.rename(video_path, video_folder + '/' + video_id + '.mp4')

        # Write the videos' name in the vid_list.csv file
        vid_list_csv.write(video_id + '.mp4\n')

    # Output all the video data to a json file
    output_file = open(output_file_name, 'w+')
    json.dump(video_list, output_file)
    vid_list_csv.close()


if __name__ == "__main__":
    process_videos(sys.argv[1], sys.argv[2], sys.argv[3])

