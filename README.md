# ActionFormer

This folder contains all the scripts and data required to work with the [**ActionFormer**](https://github.com/happyharrycn/actionformer_release) model, a transformer-based network for action detection. 

## Video metadata
The script `process-videos.py` extracts information from each video in the input folder, such as its resolution, framerate and duration, which has been written to the `video_information.json` file. Furthermore, executing it will also modify the videos' names to match the ones used in the annotation files. 

The script will also divide the dataset into a training subset and a validation one, randomly assigning each video to one of them following a chance specified as input. To call the script use:

`python process-videos.py <path to video folder> <output_file> <training_set_chance>`

## Action Annotations
To adapt the previously available annotations in the dataset, you can use the `convert-annotations.py` script, that will transform the `csv` file into two `json` annotation files, one for the verbs, and the other one for the actions. The script can be used as follows:

`python convert-annotations.py <input annotations> <video info file> <output name>`

Using this script, I translated the annotations in the file `datasetEntero_verbosYnombres.csv` (provided by Alex) to the files `video_annotations_verbs|nouns.json`, which follow the format used by **ActionFormer**.



