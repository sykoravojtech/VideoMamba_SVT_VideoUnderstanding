'''
1. Extract video frames and save it as image folder, 
2. Create a dataframe with video_id, frame_id, path, caption as the annotation file
'''
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
import os

FOLDER = 'data'
VIDEO_FOLDER = f'{FOLDER}/raw/Charades/videos'
NUM_SAMPLES_TRAIN = 200
NUM_SAMPLES_TEST = 40

train_df = pd.read_csv(f'{FOLDER}/raw/Charades/Charades_v1_train.csv')
test_df = pd.read_csv(f'{FOLDER}/raw/Charades/Charades_v1_test.csv')

if NUM_SAMPLES_TRAIN is not None:
    train_df = train_df.sample(NUM_SAMPLES_TRAIN, random_state=42)

if NUM_SAMPLES_TEST is not None:
    test_df = test_df.sample(NUM_SAMPLES_TEST, random_state=42)

################ Create a new dataframe with original_vido_id, video_id, frame_id, path, labels 
# saved to data/processed/Charades/ #######

durations = []

for df, phase in zip([train_df, test_df], ['train', 'test']):
    perframe_dict = []
    # Get the duration of the video clip
    
    # Read and save frames until the video ends
    for i, row in tqdm(df.iterrows(), total=len(df)):
        vid_code = row['id']
        vid_path = os.path.join(VIDEO_FOLDER, vid_code + '.mp4')

        # Path to the folder to save frames
        output_folder = f'{FOLDER}/processed/Charades/frames_data/{phase}/{vid_code}/'

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open the video capture
        cap = cv2.VideoCapture(vid_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        # print("fps:", fps)

        caption = row['script']

        frame_count = 0
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            if ret:
                # Save the frame as an image file
                frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_path, frame)

                perframe_dict.append({'original_vido_id': vid_code, 'video_id': "none", 
                                    'frame_id': "none", 'path': frame_path, 'caption': caption})

                frame_count += 1
            else:
                break

        # Release the video capture
        cap.release()

        durations.append(frame_count / fps)

    perframe_df = pd.DataFrame(perframe_dict)

    PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_{phase}_perframe_cap_ann.csv'
    perframe_df.to_csv(PROCESSED_CSV_PATH, index=False, sep=' ')

print("Clip durations stats:\n", pd.Series(durations).describe())

######################################################################
