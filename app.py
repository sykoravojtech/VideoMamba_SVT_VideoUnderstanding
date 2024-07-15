import streamlit as st
import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
import tempfile
from typing import Dict
from pytorchvideo.data.encoded_video import EncodedVideo
from torch import nn
from fvcore.common.config import CfgNode

from src.models import create_model
from src.datasets.transformations import get_val_transforms

DATA_DIR = "data/raw/Charades"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_VM_SETTINGS = {'config_path': 'src/config/cls_vm_charades_s224_f8_exp0.yaml',
               'weight_path': 'runs/cls_vm_ch_exp7/epoch=142-val_mAP=0.227.ckpt'}
CLS_SVT_SETTINGS = {'config_path': 'src/config/cls_svt_charades_s224_f8_exp0.yaml',
               'weight_path': '/teamspace/studios/this_studio/PracticalML_2024/runs/cls_svt_charades_s224_f8_exp0/epoch=18-val_mAP=0.165.ckpt'}
# CAP_VM_SETTINGS = {'config_path': 'src/config/cap_vm_charades_s224_f8_exp0.yaml',
#                'weight_path': 'runs/cls_vm_ch_exp7/epoch=142-val_mAP=0.227.ckpt'}
CAP_SVT_SETTINGS = {'config_path': 'src/config/cap_svt_charades_s224_f8_exp0.yaml',
               'weight_path': '/teamspace/studios/this_studio/PracticalML_2024/runs/cap_svt_charades_s224_f8_exp_32_train_all/epoch=11-step=23952.ckpt'}

@st.cache_resource()
def load_action_map():
    action_map = pd.read_csv(f"{DATA_DIR}/Charades_v1_classes_new_map.csv")
    action_label2text = action_map.set_index('label')['action'].to_dict()
    return action_label2text

class ClsModel:
    def __init__(self, config_path, weight_path):
        self.config = CfgNode.load_yaml_with_base(config_path)
        self.config = CfgNode(self.config)
        self.weight_path = weight_path
        # some constant object for inference
        self.transform = get_val_transforms(self.config)
        self.clip_duration = self.config.DATA.CLIP_DURATION
        self.num_labels = self.config.MODEL.HEAD.NUM_CLASSES

        self.lit_module = self._load_model(self.config, self.weight_path)

    @staticmethod
    def _load_model(config, weight_path):
        print('Load model')
        if config.DATA.ENCODING_DIR: # head-only weights
            lit_module = create_model(config)
            head_state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
            load_info = lit_module.load_state_dict(head_state_dict, strict=False)
        else: # load full model weights
            lit_module = create_model(config, weight_path=weight_path)

        lit_module.eval()
        
        lit_module.to(DEVICE)
        return lit_module

    def get_clip_tensors(self, video):
        """Get all clip (chunk) tensors from a video.
            Each last for 'clip_duration' seconds."""
        video_duration = float(video._duration)
        all_clip_tensors = []

        for clip_start_sec in np.arange(0, video_duration, self.clip_duration):
            clip_end_sec = min(clip_start_sec + self.clip_duration, video_duration)
            clip_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_end_sec)
            if clip_data['video'] is None:
                continue
            clip_data = self.transform(clip_data)
            clip_tensor = clip_data['video']  # (C, T, H, W)
            all_clip_tensors.append(clip_tensor)
        return all_clip_tensors

    def get_input(self, vid_path):
        """Read video, split into chunks by clip duration,
             prepare the tensors for inference."""
        video = EncodedVideo.from_path(vid_path, decode_audio=False)

        # concat clip tensors at the time dimension to write to a gif
        all_clip_tensors = self.get_clip_tensors(video)
        # Prepare video tensor for inference
        inp_video_tensors = torch.stack(all_clip_tensors, dim=0)  # (B, C, T, H, W)
        inp_video_tensors = inp_video_tensors.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        return inp_video_tensors

    def predict(self, video_path):
        inp_video_tensors = self.get_input(video_path)
        with torch.no_grad():
            inp_video_tensors = inp_video_tensors.to(DEVICE)
            model_output = self.lit_module(inp_video_tensors).sigmoid().cpu().numpy()
            model_output = model_output.max(axis=0)
        return model_output


class CapModel(ClsModel):
    def __init__(self, config_path, weight_path):
        super().__init__(config_path, weight_path)

    @staticmethod
    def _load_model(config, weight_path):
        lit_module = create_model(config)
        state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
        if config.MODEL.ENCODER.TYPE == "VideoTransformer":
            lit_module.encoder.vit.time_embed = torch.nn.Parameter(torch.zeros(1, 32, 768))
        load_info = lit_module.load_state_dict(state_dict)
        lit_module.eval().to(DEVICE)
        return lit_module


    def predict(self, video_path):
        inp_video_tensors = self.get_input(video_path)
        with torch.no_grad():
            inp_video_tensors = inp_video_tensors.to(DEVICE)
            model_output = self.lit_module.generate(inp_video_tensors, 128, 3)
        return model_output

action_label2text = load_action_map()  

def save_uploaded_file(uploaded_file, name='temp_video.mp4'):
    with open(name, 'wb') as f:
        f.write(uploaded_file.getbuffer())

def get_model(model_name, cls):
    if model_name == 'Self-supervised Video Transformer':
        if cls:
            return ClsModel(CLS_SVT_SETTINGS['config_path'], CLS_SVT_SETTINGS['weight_path'])
        else:
            return CapModel(CAP_SVT_SETTINGS['config_path'], CAP_SVT_SETTINGS['weight_path'])
    elif model_name == 'Video Mamba':
        if cls:
            return ClsModel(CLS_VM_SETTINGS['config_path'], CLS_VM_SETTINGS['weight_path'])
        else:
            return CapModel(CAP_SVT_SETTINGS['config_path'], CAP_SVT_SETTINGS['weight_path'])
    raise ValueError(f"Model {model_name} not found")


def main():
    st.title("Video Upload and Prediction")

    # Combo box for model selection
    cls_model_name = st.selectbox("Choose a model", ['Self-supervised Video Transformer', 'Video Mamba'])
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    st.video(uploaded_file)

    if uploaded_file is not None:
        print('Uploaded')
        # Save the uploaded video to a temporary file
        temp_file = '/tmp/temp_video.mp4'
        save_uploaded_file(uploaded_file, temp_file)
        
        if st.button('Predict action class'):
            cls_model = get_model(cls_model_name, True)
            model_output = cls_model.predict(temp_file)
            topk_class_indices = np.argsort(model_output)[-5:][::-1]
            text = "Top 5 classes:\n"
            for ind in topk_class_indices:
                text += f"Predicted action: {action_label2text[ind]}. Probability: {model_output[ind]:.2f}\n"

            st.text_area(label='Action classification:', value=text, height=200)
        
        if st.button("Predict caption"):
            cap_model = get_model(cls_model_name, False)
            model_output = cap_model.predict(temp_file)
            st.text("Caption: " + model_output)

if __name__ == "__main__":
    main()
