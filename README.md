# ActionFormer

This folder contains all the scripts and data required to work with the
[**ActionFormer**](https://github.com/happyharrycn/actionformer_release) model, 
a transformer-based network for action detection.

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
As the previous feature extractor did not work, we tried using a new one, [**Gluon CV**](https://cv.gluon.ai/), 
which provides feature extraction from videos using different models, including **SlowFast**. One of its main problems, 
is that if we set the number of segments of the video from which to extract features, the process will finish due
to a memory error, so it was necessary to make a script to divide each video into 32-frame long clips.

Said script can be found in `split_videos.py`, which can be executed using:

    python split_videos.py <video folder> <video clip list file> <clip size in frames>

Once the videos are split, we can extract their features using **Gluon CV**, with the following command:

    python feat_extract.py --data-list video.txt --model slowfast_4x16_resnet50_kinetics400 --save-dir ./features 
    --slowfast --slow-temporal-stride 8 --fast-temporal-stride 1 --new-length 32 --num-segments 1 --use-pretrained
    --gpu-id 1

Finally, the script `compress_features.py` is available to compress all the features extracted from individual files 
into a single one for each video. It can be executed with:

    python compress_features.py <clip feature folder> <output folder>

## SlowFast Feature Extraction

To obtain the features of our video dataset, we will use the **SlowFast** model, a video encoder based in two different
pathways, a slow one that does that processes the video in a reduced framerate, extracting semantic in formation
that is static or changes very slowly. On the other hand, the fast pathway processes the video in its original framerate,
being able to detect actions and other rapidly changing information.

To extract the features, we used an already available tool 
[**SlowFast Feature Extractor**](https://github.com/tridivb/slowfast_feature_extractor), which had to be slightly
modified as it was not up-to-date with the recent releases. In the folder `slowfast-feature-extractor` you can find 
the code necessary to execute the feature extraction.

The model was pretrained on the EpicKitchens dataset, and those weights were obtained from the projects 
[**Challenge Baselines**](https://github.com/epic-kitchens/C2-Action-Detection). 
The configuration file used was the one provided for the aforementioned dataset, modified for our own videos which should
be placed in the `datasets` folder.

To execute the code:
        
    python run_net.py --cfg <config file>

Also, it should be noted that I had to use the `resize_and_extract_frames.sh` script provided by Alex to extract each
videos frames before executing the model, as the video loading would not work otherwise. Besides, I implemented the 
`compress_numpy.py` script that compresses every `.npy` file in a given folder, using the `.npz` format 
(also required for **ActionFormer**).

## ActionFormer: Training and evaluation

The model will require two independent sessions of training, one for the nouns and another one for the verbs. To train
the model use:

    python ./train.py ./configs/bsh_verbs.yaml --output reproduce
    python ./train.py ./configs/bsh_nouns.yaml --output reproduce

And for testing:

    python ./eval.py ./configs/bsh_verbs.yaml ./ckpt/bsh_verbs_reproduce/
    python ./eval.py ./configs/bsh_verbs.yaml ./ckpt/bsh_nouns_reproduce/

The config files used are the ones provided for the EpicKitchens dataset in the ActionFormer repository, modified for
our own data.
