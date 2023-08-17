# Guillermo Enguita Lahoz 801618

# The script takes as input a file containing video annotations and another file containing each label's name and
# category. A histogram will be plotted, showing the frequency of each label in the data.

# Usage: python plot_frequencies.py <annotations file> <label info file>

import argparse
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy

colors = ['orange', 'pink', 'gold', 'cornflowerblue', 'cyan', 'lime', 'grey', 'violet', 'green', 'khaki',
          'lightsalmon', 'chocolate', 'plum', 'springgreen', 'hotpink', 'lightcoral', 'turquoise', 'moccasin',
          'mediumpurple', 'paleturquoise', 'wheat', 'saddlebrown']


def plot_histogram(annotations_file, label_info_file, args):
    # Read the annotations file and count the labels
    annotations_json = open(annotations_file)
    annotations = json.load(annotations_json)
    annotations_json.close()

    database_dict = list(annotations['database'].values())
    annotations_dict = [i['annotations'] for i in database_dict]

    annotations_dict = sum(annotations_dict, [])
    labels = [i['label'] for i in annotations_dict]

    # Load label names and categories
    label_info_csv = open(label_info_file)
    label_reader = csv.reader(label_info_csv, delimiter=',', quotechar='|')
    label_dict = {row[0]: {'name': row[1], 'category': row[2], 'frequency': 0} for row in label_reader}
    label_info_csv.close()

    # Count label frequency
    for label in labels:
        # For each label found, add 1 appearance
        label_dict[label]['frequency'] += 1

    label_list = [i for i in label_dict.values()]

    # Remove labels with 0 appearances
    label_list = [i for i in label_list if i['frequency'] > 0]

    # Sort the list by frequency and group the labels by category
    label_list.sort(key=lambda x: x['frequency'], reverse=True)
    label_list.sort(key=lambda x: x['category'], reverse=True)
    # print(label_list)

    label_names = [x['name'] for x in label_list]
    label_frequencies = [x['frequency'] for x in label_list]
    categories = [x['category'] for x in label_list]
    categories_unique = numpy.unique(numpy.array(categories))

    # Read each category's id
    range_file = open(args.CategoryRangeFile)
    csv_reader = csv.reader(range_file)
    next(csv_reader)  # Skip the header
    category_label_dict = {row[0]: int(row[1]) for row in csv_reader}

    # Get a color vector
    category_color = {category: colors[category_label_dict[category]] for category in categories_unique}
    color_list = [category_color[x['category']] for x in label_list]
    # print(color_list)

    # Plot histogram, each category will have a distinct color
    fig = plt.figure()
    xticks = [x for x in range(len(label_frequencies))]
    plt.bar(xticks, label_frequencies, color=color_list)
    plt.yticks(fontsize='20')
    plt.xticks(ticks=xticks, labels=label_names, rotation='vertical', fontsize='25')
    plt.xlim(-0.5, xticks[-1]+0.5)
    plt.title('Verb frequency', fontweight='bold', fontsize='30')
    plt.ylabel('Frequency', fontweight='bold', fontsize='25')
    plt.xlabel('Class label', fontweight='bold', fontsize='25')
    fig.set_size_inches(18.5, 8.5)

    if "hide_legend" not in args:
        # Plot legend
        legend_handles = list()

        # Create a handle for each category
        for category in reversed(category_color):
            handle = mpatches.Patch(color=category_color[category], label=category)
            legend_handles.append(handle)

        plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 20})

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog='plot_frequencies.py',
        description='The script takes as input a file containing video annotations and another file containing each '
                    'labels name and category. A histogram will be plotted, showing the frequency of each label in '
                    'the data.'
    )

    # Define the arguments needed
    parser.add_argument('AnnotationsFile', help="JSON file that contains all the annotations to process")
    parser.add_argument('LabelInfoFile', help="CSV file that contains in each row a label and its matching name and "
                                              "category")
    parser.add_argument('CategoryRangeFile', help="A csv file that contains an ID for each category, from 0 to the "
                                                      "number of categories")
    parser.add_argument('--hide_legend', action='store_true', default=argparse.SUPPRESS)

    # Parse the arguments
    args = parser.parse_args()

    plot_histogram(args.AnnotationsFile, args.LabelInfoFile, args)

