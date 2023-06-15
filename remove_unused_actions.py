# Guillermo Enguita Lahoz, 801618

"""
    This script takes as input two JSON files, which contain the action annotations of a set of videos for verbs and
    nouns respectively, as well as two files which indicate the class labels to be used, as well as their old id and
    their new one. The scrip will remove any action instance where one of the classes does not appear in the reduced
    range. Furthermore, the labels will be changed to use their new id values.
"""
import argparse
import csv
import json
import os.path


def remove_unused(noun_annotations_file, verb_annotations_file, noun_labels_file, verb_labels_file):
    # Load action annotations from json to a dict
    noun_annotations_io = open(noun_annotations_file)
    noun_annotations_json = json.load(noun_annotations_io)
    noun_database = noun_annotations_json['database']
    noun_annotations_io.close()

    verb_annotations_io = open(verb_annotations_file)
    verb_annotations_json = json.load(verb_annotations_io)
    verb_database = verb_annotations_json['database']
    verb_annotations_io.close()

    # Load label info from the csv files
    # Store it as two dicts, with old_id as key, new_id as value
    noun_labels = open(noun_labels_file, newline='')
    noun_labels_csv = csv.DictReader(noun_labels)
    noun_label_dict = dict()

    for row in noun_labels_csv:
        noun_label_dict[row['old_id']] = row['new_id']

    verb_labels = open(verb_labels_file, newline='')
    verb_labels_csv = csv.DictReader(verb_labels)
    verb_label_dict = dict()

    for row in verb_labels_csv:
        verb_label_dict[row['old_id']] = row['new_id']

    # Any videos with no annotations in the range will be removed
    # We store those videos in two lists during the loop
    nouns_empty_videos = list()
    verbs_empty_videos = list()

    # For every video in the json file
    for video_id in noun_database.keys():
        # Get the video's annotation list
        noun_annotation_list = noun_database[video_id]['annotations']
        verb_annotation_list = verb_database[video_id]['annotations']

        # Empty the list in the json file
        noun_database[video_id]['annotations'] = list()
        verb_database[video_id]['annotations'] = list()

        # For every annotation, keep it only if the label used is in the new range
        for i in range(len(noun_annotation_list)):
            noun_annotation = noun_annotation_list[i]
            verb_annotation = verb_annotation_list[i]

            # If noun label is in the new range
            if str(noun_annotation['label_id']) in noun_label_dict.keys():
                # We change the noun id to the new one and keep the interval
                new_noun_id = noun_label_dict[str(noun_annotation['label_id'])]

                noun_annotation['label_id'] = new_noun_id
                noun_annotation['label'] = new_noun_id

                noun_database[video_id]['annotations'].append(noun_annotation)

            # If the verb label is in the new range
            if str(verb_annotation['label_id']) in verb_label_dict.keys():
                # We change the verb's label and keep the interval
                new_verb_id = verb_label_dict[str(verb_annotation['label_id'])]
                verb_annotation['label_id'] = new_verb_id
                verb_annotation['label'] = new_verb_id

                verb_database[video_id]['annotations'].append(verb_annotation)

        # If there are no annotations, remove the video from the json database
        # We add the video to the empty list, and we will delete them after the loop is over
        if not noun_database[video_id]['annotations']:
            print("No noun annotations for", video_id)
            nouns_empty_videos.append(video_id)

        if not verb_database[video_id]['annotations']:
            print("No verb annotations for", video_id)
            verbs_empty_videos.append(video_id)

    # Delete any video without annotations
    for video_id in nouns_empty_videos:
        del noun_database[video_id]

    for video_id in verbs_empty_videos:
        del verb_database[video_id]

    # Output the new annotations as JSON files
    json_object_nouns = json.dumps(noun_annotations_json, indent=2)
    json_nouns_out = open(os.path.splitext(noun_annotations_file)[0] + '_reduced.json', 'w')
    json_nouns_out.write(json_object_nouns)
    json_nouns_out.close()

    json_object_verbs = json.dumps(verb_annotations_json, indent=2)
    json_verbs_out = open(os.path.splitext(verb_annotations_file)[0] + '_reduced.json', 'w')
    json_verbs_out.write(json_object_verbs)
    json_verbs_out.close()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='confusion_matrix.py',
        description='This script takes as input two JSON files, which contain the action annotations of a set of '
                    'videos for verbs and nouns respectively, as well as two files which indicate the class labels to '
                    'be used, as well as their old id and their new one. The scrip will remove any action instance '
                    'where one of the classes does not appear in the reduced range. Furthermore, the labels will be '
                    'changed to use their new id values.'
    )

    # Define the arguments needed
    parser.add_argument('NounAnnotations', help="JSON noun action annotations file")
    parser.add_argument('VerbAnnotations', help="JSON verb action annotations file")
    parser.add_argument('NounLabels', help="CSV file that indicates the nouns used, as well as their old and new ids")
    parser.add_argument('VerbLabels', help="CSV file that indicates the verbs used, as well as their old and new ids")

    # Parse the arguments
    args = parser.parse_args()
    remove_unused(args.NounAnnotations, args.VerbAnnotations, args.NounLabels, args.VerbLabels)
