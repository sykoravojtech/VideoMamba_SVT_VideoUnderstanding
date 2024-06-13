"""Use os for IO"""
import os
import pandas as pd
# import numpy as np

PATH_TO_FRAME_DATA = "/media/lav/Information Ingester/Datasets/Charades_v1_rgb/Charades_v1_rgb/"
PATH_TO_VIDEO_ANNS = "./data/raw/Charades/Charades_v1_train.csv"

def get_video_ids(path_to_frame_data):
    """Returns video ids from the per-frame rgb data in a list, given the path to the data"""
    return [d for d in os.listdir(path_to_frame_data)
            if os.path.isdir(os.path.join(path_to_frame_data, d))]

def get_frame_ids(video_id, path_to_frame_data):
    """Returns frame ids from a video of the per-frame rgb data in a list, given the path to the data and a video id"""
    video_path = os.path.join(path_to_frame_data, video_id)
    return [os.path.splitext(f)[0] for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]

# Example load and print
anns_temp = pd.read_csv(PATH_TO_VIDEO_ANNS)
print("\nFirst few lines of the video annotations:")
print(anns_temp.head())

# Example crawl and print
video_ids = get_video_ids(PATH_TO_FRAME_DATA)
print("\nA few video ids:")
print(video_ids[:10])

# Example crawl and print
print("\nA few frame ids for a few videos:")
for video_id in video_ids[:5]:
    frame_ids = get_frame_ids(video_id, PATH_TO_FRAME_DATA)
    print(f"Video ID: {video_id}, Frame IDs: {frame_ids[:5]}")