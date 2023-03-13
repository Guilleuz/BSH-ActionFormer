# Guillermo Enguita Lahoz 801618

# This script takes as input the action annotations for the EpicKitchens dataset and our own dataset and fuses both of
# them in the same annotation files, using the EpicKitchens one as the training set and the BSH videos as the validation
# subset. Execute with:

# python fuse_annotations.py <epic's noun annotations> <epic's features folder> <epic's verb annotations>
#   <bsh's noun annotations> <bsh's verbs annotations> <bsh's feature folder>

# Furthermore, the feature file names will be changed to distinguish them from one another, adding the prefix "EK_" to
# Epic Kitchens ones and "BSH_" to our own, both in the files and in the annotations.


def fuse_annotations(epic_noun_file, epic_verb_file, bsh_noun_file, bsh_verb_file):


def change_video_names():