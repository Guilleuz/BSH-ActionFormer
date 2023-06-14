# Guillermo Enguita Lahoz, 801618

"""
    This script takes as input an action annotation files, and outputs the same file with a reduced label range.
    That is, the original labels will be given a new id ranging from 0 to number of labels - 1.
    It will also output two csv files with the label names, their original ids and their new ids.
"""

import argparse
import csv
import os


# Takes as input an action annotations csv file and modifies the label ids, so they range from 0 to number of labels - 1
# Also outputs two new files, one with the label ids for nouns, and the other for verbs
def reduce_label_range(annotations_file_path):
    # Create the csv reader
    annotations_file = open(annotations_file_path)
    csv_reader = csv.DictReader(annotations_file, delimiter=',')

    # Create the csv writer for the output file
    # Set column headers
    file_headers = ['narration_id', 'video_id', 'start_timestamp', 'stop_timestamp', 'narration', 'verb', 'verb_class',
                    'noun', 'noun_class']

    # Set output file path
    output_path = os.path.splitext(annotations_file_path)[0] + '_reduced.csv'
    output_file = open(output_path, 'w', newline='')

    # Create writer and write headers
    writer = csv.DictWriter(output_file, fieldnames=file_headers)
    writer.writeheader()

    # Dicts to save each label and its old/new id
    noun_label_ids = dict()
    verb_label_ids = dict()

    # Counters for the next new ids
    noun_label_counter = 0
    verb_label_counter = 0

    # Read every row in the file
    for row in csv_reader:
        # If the label is new, we save it and give it a new id
        noun_label = row['noun']
        verb_label = row['verb']

        noun_id = row['noun_class']
        verb_id = row['verb_class']

        if noun_id not in noun_label_ids.keys():
            noun_label_ids[noun_id] = {'label_name': noun_label, 'new_label': noun_label_counter}
            noun_label_counter += 1

        if verb_id not in verb_label_ids.keys():
            verb_label_ids[verb_id] = {'label_name': verb_label, 'new_label': verb_label_counter}
            verb_label_counter += 1

        # Change the label's id
        row['noun_class'] = noun_label_ids[noun_id]['new_label']
        row['verb_class'] = verb_label_ids[verb_id]['new_label']

        # Write the modified row to the output file
        writer.writerow(row)

    annotations_file.close()
    output_file.close()

    # Save label names, their old ids and the new ones to file
    save_label_ids(noun_label_ids, 'noun')

    save_label_ids(verb_label_ids, 'verb')


# Saves a dictionary of label ids (of type 'noun' or 'verb') to a csv file
def save_label_ids(label_ids_dict, label_type):
    output_file = open(label_type + '_labels.csv', 'w', newline='')
    output_writer = csv.DictWriter(output_file, fieldnames=['label_name', 'old_id', 'new_id'])

    output_writer.writeheader()

    for label_id in label_ids_dict.keys():
        print(label_ids_dict[label_id])
        output_writer.writerow({'label_name': label_ids_dict[label_id]['label_name'], 'old_id': label_id,
                                'new_id': label_ids_dict[label_id]['new_label']})


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='confusion_matrix.py',
        description='This script takes as input a CSV action annotation files, and outputs the same file with a '
                    'reduced label range. That is, the original labels will be given a new id ranging from 0 to '
                    'number of labels - 1. It will also output two csv files with the label names, their original ids '
                    'and their new ids.'
    )

    # Define the arguments needed
    parser.add_argument('AnnotationsFile', help="CSV action annotations file")

    # Parse the arguments
    args = parser.parse_args()
    reduce_label_range(args.AnnotationsFile)


