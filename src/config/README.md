# Naming scheme:
`<task>_<backbone>_<dataset>_<img_size>_<num_sampled_frames>_<experiment_num>.yaml`

# Intro
PROJECT_NAME: PracticalML
- doesn't need to be changed

EXPERIMENT: cls_svt_ucf101_s224_f8_exp0
- change depending on current run
- check lower for full explanation of namescheme

OUTPUT_DIR: ./runs
- output directory for checkpoint .ckpt files

WANDB_KEY: ___
- weights and biases key

SEED: 42
- Seed for reproducibility

# MODEL
TYPE: `_factory.py:create_model` 
- if == 'classification'
    - VideoClassificationModel 
- if == 'captioning'
    - VideoCaptioningnModel

LOSS: 
- CrossEntropyLoss (cap)
- BCEWithLogitsLoss (cls)

## ENCODER
TYPE: `_encoder_factory.py:create_encoder` 
- if == 'VideoTransformer':
    - VideoTransformerEncoder
- if == 'VideoMamba':
    - VideoMambaEncoder

## HEAD
TYPE: `_head_factory.py:create_head`
- if == 'MLP'
    - MLPHead()
- if == 'Generative':
    - GenerativeHead

# DATA
DATASET: charades_caption
- `datasets/_factory.py` chooses which dataset to load

ROOT_PATH: data/raw/Charades
- root path of the dataset

TRAIN_CSV: Charades_per-frame_annotations_captioning_train.csv
- csv file containing links to train set

TEST_CSV: Charades_per-frame_annotations_captioning_test.csv
- csv file containing links to test set

IMG_SIZE: 224
- image size

NUM_SAMPLED_FRAMES: 8
- how many frames we want sampled

CLIP_DURATION: 30

MEAN: [0.45, 0.45, 0.45]

STD: [0.225, 0.225, 0.225]

NUM_WORKERS: 16

# TRAIN
FREEZE_ENCODER: True

NUM_EPOCHS: 30

BATCH_SIZE: 32

LOG_STEPS: 50

ACCELERATOR: auto

DEVICES: auto

PRECISION: 16

BEST_CHECKPOINT_BY: val_mAP

COMPUTE_METRIC_AT_TRAIN_TIME: True

## OPTIM
TYPE: AdamW

INIT_LEARNING_RATE: 0.0001

MIN_LEARNING_RATE: 0.000001

EPS: 0.000001

BETAS: [0.9, 0.999]

LR_MILESTONES: [25]
- at which epochs should we multiply learning rate by LR_GAMMA

LR_GAMMA: 0.1

# ENCODERS
## VideoTransformer
PRETRAINED: checkpoints/kinetics400_vitb_ssl.pth
HIDDEN_SIZE: 768
RETURN_ALL_HIDDEN: False

## VideoMamba
PRETRAINED: if this exists, the models know the path to the checkpoints
MODEL_SIZE:  choose from   tiny, small, middle
HIDDEN_SIZE: corresponding 192,  384,   576

# HEADS
## MLP
LAYERS: [128,128]
NUM_CLASSES: 10

## Generative
LANGUAGE_MODEL: distilgpt2

MAX_TOKENS: 128