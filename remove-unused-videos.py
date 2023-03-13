# Guillermo Enguita Lahoz 801618
# Script to delete all of the dataset videos that are not annotated
# It takes as imput a video folder, and will remove any video that is not listed in the vid_list.csv
# The file vid_list.csv must exist in the folder

# Execute using: python remove-unused-videos.py <video folder>

import os
import sys


# Removes any video in the video_folder path that is not listed in the vid_list.csv file
def remove_unused_videos(video_folder):
    vid_list_file = open(video_folder + '/vid_list.csv')
    annotated_videos = vid_list_file.readlines()

    # Strip the newline characters
    for i in range(0, len(annotated_videos)):
        annotated_videos[i] = annotated_videos[i].strip()

    # Check all the videos in the folder, and delete them if necessary
    for video in os.listdir(video_folder):
        if video not in annotated_videos and video != 'vid_list.csv':
            print(video, "removed")
            os.remove(video_folder + '/' + video)


if __name__ == "__main__":
    remove_unused_videos(sys.argv[1])
