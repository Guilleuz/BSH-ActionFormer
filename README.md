# ActionFormer

This folder contains all the scripts and data required to work with the
[**ActionFormer**](https://github.com/happyharrycn/actionformer_release) model, 
a transformer-based network for action detection.

## Folder structure

### *Code* folder

Contains all the code developed until now. *ActionFormer's* repository is stored in the `ActionFormer` subdirectory, the
code has been slightly modified, mainly to output the intervals predicted by the model.

The `Scripts` subdirectory includes all the auxiliary python scripts needed to process the input data, so that our videos
can be used with _ActionFormer_, as well as feature extraction and interval plotting, among others. 

### *Data* folder

Contains all the action annotations for BSH's and Epic Kitchens videos as well as the features used as input for the
model. The folder `video_annotaions` contains some extra annotations for the dataset, mainly a json file with detailed
information of each video, two files which contain the class names associated to each of the labels, the original action
annotations and a list with all the video names to process.

Finally, the `output_intervals` folder has some of the intervals predicted by the model, which can be used as input for the plot script.


## Video metadata
The script `process-videos.py` extracts information from each video in the input folder, such as its resolution, 
framerate and duration, which has been output to the `video_information.json` file. Furthermore, executing it will
also modify the videos' names to match the ones used in the annotation files. 

The script will also divide the dataset into a training subset and a validation one, randomly assigning each video to 
one of them following a chance specified as input. To call the script use:

    python process-videos.py <path to video folder> <output_file> <training_set_chance>

## Action Annotations
To adapt the previously available annotations in the dataset, you can use the `convert-annotations.py` script, that will
transform the `csv` file into two `json` annotation files, one for the verbs, and the other one for the actions.
The script can be used as follows:

    python convert-annotations.py <input annotations> <video info file> <output name>

Using this script, I translated the annotations in the file `datasetEntero_verbosYnombres.csv` (provided by Alex) to 
the files `video_annotations_verbs|nouns.json`, which follow the format used by **ActionFormer**. It also outputs the
file `vid_list.csv`, which contains a list of the annotated videos, and is required for the 
**SlowFast Feature Extractor**. This file has to be included in the same folder as the videos whose features are going
to be extracted.

Furthermore, the `remove-unused-videos.py` script will delete (do NOT execute if you still need them) any videos from 
the video folder that have not been annotated, that is, videos not listed in `vid_list.csv`.


## Gluon CV
For extracting the features we use the [**Gluon CV**](https://cv.gluon.ai/) tool, 
which provides feature extraction from videos using different models, including **SlowFast**. One of its main problems, 
is that if we set the number of segments of the video from which to extract features, the process will finish due
to a memory error, so it was necessary to make a script to divide each video into 32-frame long clips.

Said script can be found in `split_videos.py`, which can be executed using:

    python split_videos.py <video folder> <video clip list file> <clip size in frames>

Once the videos are split, we can extract their features using **Gluon CV** (it requires installation), 
with the following command:

    python feat_extract.py --data-list video.txt --model slowfast_4x16_resnet50_kinetics400 --save-dir ./features 
    --slowfast --slow-temporal-stride 8 --fast-temporal-stride 1 --new-length 32 --num-segments 1 --use-pretrained
    --gpu-id 1

Finally, the script `compress_features.py` is available to compress all the features extracted from individual files 
into a single one for each video. It can be executed with:

    python compress_features.py <clip feature folder> <output folder>

## ActionFormer: Training and evaluation

The model will require two independent sessions of training, one for the nouns and another one for the verbs. To train
the model use:

    python ./train.py ./configs/bsh_verbs.yaml --output reproduce
    python ./train.py ./configs/bsh_nouns.yaml --output reproduce

And for testing:

    python ./eval.py ./configs/bsh_verbs.yaml ./ckpt/bsh_verbs_reproduce/
    python ./eval.py ./configs/bsh_verbs.yaml ./ckpt/bsh_nouns_reproduce/

The config files used are the ones provided for the EpicKitchens dataset in the ActionFormer repository, modified for
our own data. After the testing, the model will output two csv files `ground_truth.csv` and `preds.csv`, which contain
the original action intervals and the predicted ones.

## Prediction Results
To better visualize the results obtained after inference, we provide the `show_predictions.py` script, which will plot a
graph showing the predicted action intervals and the actual ones. To run it, we will need the `ground_truth.csv` and 
`preds.csv` files obtained after evaluation. 

To run the script, execute:
    
    python show_predictions.py --ground_truth <ground csv file> --predictions <preds csv file> --threshold <value> --separated

Furthermore, using the option `--help` will show all the available options. If the flag `--web` is set, and you run with 
`streamlit run show_predictions.py -- [ARGS]`, the plot will generate an interactive html graph, using `streamlit`.

![](/home/guille/PycharmProjects/BSH-ActionFormer/example_plot.png)
