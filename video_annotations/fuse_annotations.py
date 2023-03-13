# Guillermo Enguita Lahoz 801618

# This script takes as input the action annotations for the EpicKitchens dataset and our own dataset and fuses both of
# them in the same annotation files, using the EpicKitchens one as the training set and the BSH videos as the validation
# subset. Execute with:

# python fuse_annotations.py <epic's noun annotations> <epic's features folder> <epic's verb annotations>
#   <bsh's noun annotations> <bsh's verbs annotations> <bsh's feature folder>

# Furthermore, it offers a function to add a prefix to all files in a folder

import json
import os
import sys


# Combines two annotation files, one from Epic Kitchens and the other from our dataset
def combine_annotations(bsh_file, epic_file, output_file):
    # Load noun annotations
    epic_file_f = open(epic_file)
    epic_dataset = json.load(epic_file_f)

    bsh_file_f = open(bsh_file)
    bsh_dataset = json.load(bsh_file_f)

    # Change every annotation in the Epic Kitchens dataset to the training subset
    for entry in epic_dataset['database']:
        epic_dataset['database'][entry]['subset'] = 'training'

    # Change every annotation in our dataset to the validation subset, and add it to Epic's annotation list
    for entry in bsh_dataset['database']:
        bsh_dataset['database'][entry]['subset'] = 'validation'
        epic_dataset['database']['BSH_' + entry] = bsh_dataset['database'][entry]

    # Output the combined annotations
    output = open(output_file, 'w+')
    json.dump(epic_dataset, output)


# Combines noun and verb annotations
def fuse_annotations(epic_noun_file, epic_verb_file, bsh_noun_file, bsh_verb_file):
    combine_annotations(bsh_noun_file, epic_noun_file, 'noun_annotations.json')
    combine_annotations(bsh_verb_file, epic_verb_file, 'verb_annotations.json')


# Adds the prefix 'prefix' to all the files found in 'folder'
def change_file_names(folder, prefix):
    for file in os.listdir(folder):
        os.rename(folder + '/' + file, folder + '/' + prefix + file)


def main():
    fuse_annotations(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
