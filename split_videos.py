# Guillermo Enguita Lahoz 801618
# This script splits all videos in a specified folder in clips of length given by clip_size.
# All the clips will be stored in new directories created after each video, and their paths will be specified in the
# video clip list file.
# Execute with:
# python split_videos.py <video folder> <video clip list file> <clip size in frames> <frame stride>

from tqdm import tqdm
import glob
import os
import sys
from math import floor
import cv2


# Takes all the videos in a folder and splits them into clips of clip_size length
def split_videos(video_folder, video_list_file, clip_size, stride):
    # Process args
    clip_size = int(clip_size)
    stride = int(stride)
    video_list = open(video_list_file, 'w+')

    # List all videos
    videos = glob.glob(video_folder + '/*.mp4')

    video_num = 1

    # For every video in the folder
    for video in videos:
        # Create the folder for the clips, if needed
        video_name = os.path.basename(video)
        video_name = os.path.splitext(video_name)[0]
        folder_name = video_name + '_clips'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        print('Processing ', video_name, ' (', video_num, '/', len(videos), ')', sep='')
        video_num += 1

        # Instantiate the video capture for the current video
        video_capture = cv2.VideoCapture(video)

        # Get number of frames and number of clips
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        num_clips = floor(total_frames / stride)
        bar = tqdm(range(num_clips))
        frames = []

        # Read the first frame
        success, frame = video_capture.read()
        clip_num = 1

        # Read loop
        while success:
            frames.append(frame)

            # Write the clip if the number of read frames is equal to the clip size
            if len(frames) == clip_size:
                rows, cols, _ = frames[0].shape
                out = cv2.VideoWriter(folder_name + '/' + str(clip_num) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                      (cols, rows))
                for frame in frames:
                    out.write(frame)
                out.release()
                # frames = []
                frames = frames[stride - 1:-1]

                # Write the clip path to the video clip file
                video_list.write(folder_name + '/' + str(clip_num) + '.mp4\n')
                bar.update(1)
                bar.refresh()
                clip_num += 1

            # Read the next frame
            success, frame = video_capture.read()

        # Write the last few frames, adding padding to create a full clip
        if len(frames) != 0:
            # Repeat the last frame the necessary times
            last_frame = frames[len(frames) - 1]
            padding_size = clip_size - len(frames)
            padding = [last_frame] * padding_size
            frames.extend(padding)

            # Save the clip
            rows, cols, _ = frames[0].shape
            out = cv2.VideoWriter(folder_name + '/' + str(clip_num) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                  (cols, rows))

            for frame in frames:
                out.write(frame)
            out.release()
            bar.update(1)
            bar.refresh()

            video_list.write(folder_name + '/' + str(clip_num) + '.mp4\n')

    video_list.close()


if __name__ == "__main__":
    print(cv2.getBuildInformation())
    split_videos(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
