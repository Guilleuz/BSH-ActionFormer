# Guillermo Enguita Lahoz 801618
# The script takes as input a folder path, and will compress any .npy files in that folder to the .npz format
# Execute with: python compress_numpy.py <path to folder>

import sys
import os
import numpy as np


# Takes all the npy files from a given folder and compresses them to the npz format
def compress_npy(npy_folder):
    # For all files in the folder
    for file in os.listdir(npy_folder):
        # Check if they are on .npy format
        if file.endswith('.npy'):
            # Load the file and compress as .npz
            print('Compressing ', file, '...', end='', sep='')
            array = np.load(npy_folder + '/' + file)
            basename = os.path.splitext(file)[0]

            # Remove the '_32' suffix given by SlowFast (if present)
            basename_splits = basename.split(sep='_')
            basename_splits = basename_splits[0:len(basename_splits)-1]
            if len(basename_splits) == 2:
                basename = '_'.join(basename_splits)

            np.savez_compressed(npy_folder + '/' + basename + '.npz', feats=array)
            print(' Done!')

def change_label(npz_folder):
    for npz_file in os.listdir(npz_folder):
        with open(npz_folder + '/' + npz_file, "rb") as npy:
            print('Changing label for', npz_file)
            array = np.load(npy)            
            if array.files[0] == 'arr_0':
                np.savez_compressed(npz_folder + '/' + npz_file, feats=array['arr_0'])
                print('Done!')
            else:
                print("Skipped")
            


if __name__ == '__main__':
    change_label(sys.argv[1])
    # compress_npy(sys.argv[1])
