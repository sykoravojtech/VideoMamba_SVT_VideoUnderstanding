
import pandas as pd
import numpy as np
import cv2
import os

FOLDER = 'data'

df = pd.read_csv(f'{FOLDER}/raw/Charades/Charades_v1_train.csv')

##### Create the action labels csv file: convert action codes to integer labels. 
# Save to: data/processed/Charades/Charades_v1_classes.csv #####
ACTIONT_TXT = f'{FOLDER}/raw/Charades/Charades_v1_classes.txt'
action_ids = []
actions = []
for row in open(ACTIONT_TXT):
    row = row.split(' ')
    action_ids.append(row[0])
    actions.append(' '.join(row[1:]).strip())
    # break

action_df = pd.DataFrame({'action_id': action_ids, 'action': actions})
action_df['label'] = np.arange(len(action_df))
os.makedirs(f'{FOLDER}/processed/Charades', exist_ok=True)
action_df.to_csv(f'{FOLDER}/processed/Charades/Charades_v1_classes.csv', index=False)
######################################################################

##### Supporting function to convert action codes to integer labels #####
action_id_to_label = action_df.set_index('action_id')['label'].to_dict()

def get_label(str_labels):
    '''Convert a sequence of (start time, end time, action id) to a sequence of integer labels.'''
    int_labels = []
    if pd.isnull(str_labels):
        return ''
    for item in str_labels.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(str(action_id_to_label[action_id]))
    return "," .join(int_labels)

VIDEO_FOLDER = f'{FOLDER}/raw/Charades/videos'
####################################################


################ Create a new dataframe with original_vido_id, video_id, frame_id, path, labels 
# as required by Charades dataset, saved to data/processed/Charades/Charades_v1_train_pytorchvideo_perframe.csv #######

demo_vid_code = df['id'].iloc[0] # only demo for 1 video
demo_vid_path = os.path.join(VIDEO_FOLDER, demo_vid_code + '.mp4')
demo_vid_labels = get_label(df['actions'].iloc[0]) ### ex: "2,3,4" (all the actions for the entire video)

perframe_dict = []

# Path to the folder to save frames
output_folder = f'{FOLDER}/processed/Charades/frames_data/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video capture
cap = cv2.VideoCapture(demo_vid_path)

# Read and save frames until the video ends
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)

        perframe_dict.append({'original_vido_id': demo_vid_code, 'video_id': "none", 
                            'frame_id': "none", 'path': frame_path, 'labels': demo_vid_labels})

        frame_count += 1
    else:
        break

# Release the video capture
cap.release()

perframe_df = pd.DataFrame(perframe_dict)

PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_train_pytorchvideo_perframe.csv'
perframe_df.to_csv(PROCESSED_CSV_PATH, index=False, sep=' ')
######################################################################

