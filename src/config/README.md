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

## ENCODER
TYPE: `_encoder_factory.py:create_encoder` 
- if == 'VideoTransformer':
    - VideoTransformerEncoder
- if == 'VideoMamba':
    - VideoMambaEncoder

## HEAD
TYPE:
- if == 'MLP'
    - MLPHead()

# DATA

# TRAIN

## OPTIM

# ENCODERS
## VideoTransformer
PRETRAINED: checkpoints/kinetics400_vitb_ssl.pth
HIDDEN_SIZE: 768
RETURN_ALL_HIDDEN: False

# HEADS
## MLP
LAYERS: [128,128]
NUM_CLASSES: 10