# Informed methods for source speaker verification

**Original author:** Vojtěch Staněk ([vojteskas](https://github.com/vojteskas)), xstane45@vutbr.cz

**Fork contributor:** Zbyněk Lička ([pandavinci](https://github.com/pandavinci)) ilicka@vutbr.cz

This repository is a fork of my colleagues Master's degree implementation. I extend it from deepfake detection to source speaker verification.

## Requirements
This repository comes with a Dockerfile:
```bash
# build the environment:
docker build -t $IMAGE_NAME
# or use a prepackaged one:
docker import example_image.tar.gz $IMAGE_NAME
# run an interactive environment
docker run -it -v $DATASETS_PATH/$DATASET:/datasets/$DATASET -v $REPOSITORY_PATH:/app $IMAGE_NAME:latest
```
if you use an interactive docker environment this way, the trained models will be saved in `$REPOSITORY_PATH/checkpoints` by default.

**Python 3.10**, possibly works with newer versions\
**PyTorch >2.2.0** including torchvision and torchaudio \
packages in `requirements.txt`

Simply install the required conda environment with:

```
# optional, create and activate conda env
# conda create -n diff_detection python=3.10
# conda activate diff_detection

# install required packages
# !!always refer to pytorch website https://pytorch.org/ for up-to-date command!!
# conda install pytorch torchvision torchaudio cpuonly -c pytorch  # For CPU-only install
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  # For GPU-enabled install

pip install -r requirements.txt
```

## Usage

Based on the use-case, use either `train_and_eval.py` or `eval.py` scripts with the following arguments:

```
usage:
train_and_eval.py [-h] (--metacentrum | --sge | --local) [--checkpoint CHECKPOINT] -d DATASET -e EXTRACTOR -p {MHFA,AASIST,Mean,SLS} [--loss {CrossEntropy,AdditiveAngularMargin}] [--margin MARGIN] [--s S] [--easy_margin] -c CLASSIFIER [-a] [--n_components N_COMPONENTS]
                         [--covariance_type COVARIANCE_TYPE] [--kernel KERNEL] [-ep NUM_EPOCHS] [--sampling SAMPLING]

Main script for training and evaluating the classifiers.

options:
  -h, --help            show this help message and exit
  --metacentrum         Flag for running on metacentrum.
  --sge                 Flag for running on SGE on BUT FIT.
  --local               Flag for running locally.
  --checkpoint CHECKPOINT
                        Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.
  -d DATASET, --dataset DATASET
                        Dataset to be used. See common.DATASETS for available datasets.
  -e EXTRACTOR, --extractor EXTRACTOR
                        Extractor to be used. See common.EXTRACTORS for available extractors.
  -p {MHFA,AASIST,Mean,SLS}, --processor {MHFA,AASIST,Mean,SLS}, --pooling {MHFA,AASIST,Mean,SLS}
                        Feature processor/pooling to be used. Options: ['MHFA', 'AASIST', 'Mean', 'SLS']
  --loss {CrossEntropy,AdditiveAngularMargin}
                        Loss function to be used. Options: ['CrossEntropy', 'AdditiveAngularMargin']. Default: CrossEntropy.
  --margin MARGIN       Margin parameter for AdditiveAngularMargin loss (default: 0.5)
  --s S                 Scale parameter for AdditiveAngularMargin loss (default: 30.0)
  --easy_margin         Use easy margin for AdditiveAngularMargin loss
  -c CLASSIFIER, --classifier CLASSIFIER
                        Classifier to be used. See common.CLASSIFIERS for available classifiers.
  -a, --augment         Flag for whether to use augmentations during training. Does nothing during evaluation.

Classifier-specific arguments:
  --n_components N_COMPONENTS
                        n_components for GMMDiff
  --covariance_type COVARIANCE_TYPE
                        covariance_type for GMMDiff
  --kernel KERNEL       kernel for SVMDiff. One of: linear, poly, rbf, sigmoid
  -ep NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to train for. Does not concern SkLearn classifiers.
  --sampling SAMPLING   Variant of sampling the data for training SkLearn mocels. One of: all, avg_pool, random_sample.
```

## Repository structure

```
DP
├── augmentation        <- contains various data augmentation techniques
├── classifiers         <- contains the classes for models
│   ├── differential        <- pair-input
│   └── single_input        <- single-input
├── datasets            <- contains Dataset classes (ASVspoof (2019, 2021), ASVspoof5, In-the-Wild, Morphing)
├── extractors          <- contains various feature extractors
├── feature_processors  <- contains pooling implementation (avg pool, MHFA, AASIST)
├── scripts             <- output directory for script_generator.py
├── trainers            <- contains classes for training and evaluating models
├── losses              <- contains loss functions 
├ Makefile
├ README.md
├ common.py             <- common code, enums, maps, dataloaders
├ config.py             <- hardcoded config, paths, batch size
├ eval.py               <- script for evaluating trained model
├ parse_arguments.py    <- argument parsing script
├ requirements.txt      <- requirements to install in conda environment
├ runner.sh             <- script for simultaneously running scripts from scripts folder
├ scores_utils.py       <- functions for score analysis and evaluation
├ script_generator.py   <- helper script to generate job scripts for metacentrum
└ train_and_eval.py     <- main script for training and evaluating models
```

## Publications

Here, the related publications will be listed.

Rohdin, J., Zhang, L., Oldřich, P., Staněk, V., Mihola, D., Peng, J., Stafylakis, T., Beveraki, D., Silnova, A., Brukner, J., Burget, L. (2024) *BUT systems and analyses for the ASVspoof 5 Challenge*. Proc. The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024), 24-31, DOI: [10.21437/ASVspoof.2024-4](https://www.isca-archive.org/asvspoof_2024/rohdin24_asvspoof.html)

## Contact

For any inquiries, questions or ask for help/explanation, contact me at ilicka@vutbr.cz.
