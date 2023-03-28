# Guillermo Enguita Lahoz 801618

# The script takes all the clip feature folders inside 'feature_folder', loads all the features obtained from each
# individual clip and compresses it to a single file, saved with the original videos name + '.npz' to the
# 'output_folder' path.

# Execute with: python compress_features.py <clip feature folder> <output folder>

import os
import sys
from glob import glob

import numpy as np
from tqdm import tqdm


# Processes all the clip folders inside of feature_folder. For each one of them, we extract all the clips' features,
# concatenate them in a single numpy array and save it with the original video's name (.npz) in the output_folder
def compress_features(feature_folder, output_folder):
    # If the output_folder does not exist, we create it
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # For each video folder inside the feature folder
    for video_folder, _, _ in os.walk(feature_folder):
        if video_folder == feature_folder:
            continue

        # Get the original name of the video
        video_id = '_'.join(os.path.basename(video_folder).split('_')[0:2])
        print('Compressing ', video_id, '\'s features...', sep='')

        # Read all the clips, sort them by numerical order
        clip_list = map(os.path.basename, glob(video_folder + '/*.npz'))
        clip_list = sorted(clip_list, key=lambda x: int(x.split('.')[0]))
        bar = tqdm(range(len(clip_list)), leave=False)

        # Empty array
        array = np.empty((0, 2304), dtype=np.float32)

        # For each clip, we concatenate it in the array
        for clip_name in clip_list:
            # Load clip
            row = np.asarray(np.load(video_folder + '/' + clip_name)['arr_0'])
            row = row.flatten()

            # Concatenate
            array = np.vstack([array, row])

            bar.update(1)
            bar.refresh()

        # Save the compressed feature array to the output folder using the video id
        print('\nDone')
        np.savez_compressed(output_folder + '/' + video_id + '.npz', feats=array)


if __name__ == "__main__":
    compress_features(sys.argv[1], sys.argv[2])
