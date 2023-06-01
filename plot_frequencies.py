# Guillermo Enguita Lahoz 801618

# The script takes as input a file containing video annotations and another file containing each label's name and
# category. A histogram will be plotted, showing the frequency of each label in the data.

# Usage: python plot_frequencies.py <annotations file> <label info file>

import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy

colors = ['orange', 'pink', 'gold', 'cornflowerblue', 'cyan', 'lime', 'grey', 'violet', 'palegreen', 'khaki',
          'lightsalmon', 'chocolate', 'plum', 'springgreen', 'hotpink', 'lightcoral', 'turquoise', 'moccasin',
          'mediumpurple', 'paleturquoise', 'wheat', 'saddlebrown']


def plot_histogram(annotations_file, label_info_file):
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
    print(label_list)

    label_names = [x['name'] for x in label_list]
    label_frequencies = [x['frequency'] for x in label_list]
    categories = [x['category'] for x in label_list]
    categories_unique = numpy.unique(numpy.array(categories))

    # Get a color vector
    category_color = {categories_unique[i]: colors[i] for i in range(len(categories_unique))}
    color_list = [category_color[x['category']] for x in label_list]
    print(color_list)

    # Plot histogram, each category will have a distinct color
    xticks = [x for x in range(len(label_frequencies))]
    plt.bar(xticks, label_frequencies, color=color_list)
    plt.xticks(ticks=xticks, labels=label_names, rotation='vertical')
    plt.xlim(-0.5, xticks[-1]+0.5)
    plt.title('Class frequency (Nouns)')
    plt.ylabel('Frequency')
    plt.xlabel('Class label')

    # Add category text
    # May not be implemented, the text boxes overlap as there is very little room between categories
    """previous = None
    for i, category in enumerate(categories):
        if category != previous:
            previous = category
            plt.text(xticks[i], 100, category.capitalize(), rotation='vertical',
                     bbox=dict(facecolor=category_color[category], alpha=0.5))"""

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

    # Parse the arguments
    args = parser.parse_args()

    plot_histogram(args.AnnotationsFile, args.LabelInfoFile)

