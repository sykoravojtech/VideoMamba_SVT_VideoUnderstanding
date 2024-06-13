"""Use os for IO"""
import os
import pandas as pd
# import numpy as np

PATH_TO_RGB_DATA = "/media/lav/Information Ingester/Datasets/Charades_v1_rgb/Charades_v1_rgb/"
PATH_TO_VIDEO_ANNS = "./data/raw/Charades/Charades_v1_train.csv"

def get_video_ids(path_to_rgb_data):
    """Gathers video ids from the per-frame rgb data in a list, given the path to that data"""
    return [d for d in os.listdir(path_to_rgb_data)
            if os.path.isdir(os.path.join(path_to_rgb_data, d))]

# Example load and print
anns_temp = pd.read_csv(PATH_TO_VIDEO_ANNS)
print("\nExample load and print:")
print(anns_temp.head())

# Example crawl and print
video_ids = get_video_ids(PATH_TO_RGB_DATA)
print("\nExample crawl and print:")
print(video_ids[:10])
