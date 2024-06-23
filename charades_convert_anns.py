"""Use os for IO"""
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

"""
This script takes charades video annotations from ../Charades/Charades_v1_{train/test}.csv 
and frame data from ../Charades/Charades_frames/ and extrapolates per-frame
annotations with per-frame video labels. The resulting annotations are stored in
../Charades/Charades_per-frame_annotations_{train/test}.csv
"""

PATH_TO_CHARADES_ROOT = "./data/raw/Charades/"
PATH_TO_FRAME_DATA = f"./data/raw/Charades_frames/Charades_v1_rgb/"
# adjust file path

DUMMY_1 = -1  # placeholder for video_id, unused in the implementation
DUMMY_2 = -1  # placeholder for frame_id, unused in the implementation

FPS = 24  # As per https://prior.allenai.org/projects/charades set fps to 24

MULTILABEL = False  # Toggle mutlilabel vs single label format.

# Create the action labels csv file: convert action codes to integer labels.
# Save to: .../Charades/Charades_v1_classes_new_map.csv
ACTION_TXT = f'{PATH_TO_CHARADES_ROOT}/Charades_v1_classes.txt'

action_ids = []
actions = []
for row in open(ACTION_TXT):
    row = row.split(' ')
    action_ids.append(row[0])
    actions.append(' '.join(row[1:]).strip())
    # break

action_df = pd.DataFrame({'action_id': action_ids, 'action': actions})
action_df['label'] = np.arange(len(action_df))
action_df.to_csv(f'{PATH_TO_CHARADES_ROOT}/Charades_v1_classes_new_map.csv', index=False)



# Create supporting dict to convert action codes to integer labels
ACTION_ID_TO_LABEL = action_df.set_index('action_id')['label'].to_dict()


def get_video_ids(path_to_frame_data):
    """Returns video ids from the per-frame rgb data in a list,
            given the path to the data"""
    return [d for d in os.listdir(path_to_frame_data)
            if os.path.isdir(os.path.join(path_to_frame_data, d))]

def get_frame_ids(vid_id, path_to_frame_data):
    """Returns frame ids from a video of the per-frame rgb data in a list, 
            given the path to the data and a video id"""
    video_path = os.path.join(path_to_frame_data, vid_id)
    return [os.path.splitext(f)[0] for f in os.listdir(video_path)
            if os.path.isfile(os.path.join(video_path, f))]

def get_labels(str_labels):
    """Convert a sequence of (start time, end time, action id) to a sequence of integer labels.
            Returns a 1d list of integers."""
    int_labels = []
    if pd.isnull(str_labels):
        return ''
    for item in str_labels.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(ACTION_ID_TO_LABEL[action_id])
    return int_labels

def create_frame_anns(vid_anns, path_to_frame_data):
    """Returns annotations with the desired frame paths,
            given the path to the data and the full video ids"""
    frm_anns = []
    for _, anns_row in tqdm(vid_anns.iterrows(), total=len(vid_anns)):
        vid_id = anns_row['id']
        frm_ids = get_frame_ids(vid_id, path_to_frame_data)
        if MULTILABEL:
            int_labels = get_labels(anns_row['actions']) if pd.notnull(anns_row['actions']) else ''
            str_multilabel = []
            for int_label in int_labels:
                str_multilabel.append(int_label)
            str_multilabel = "," .join(str_multilabel)

            for frm_id in frm_ids:
                frm_path = os.path.join(path_to_frame_data, vid_id, f"{frm_id}.jpg")
                ann_entry = (vid_id, DUMMY_1, DUMMY_2, frm_path, str_multilabel)
                frm_anns.append(ann_entry)

        if not MULTILABEL:
            # separate video instance into multiple action chunks, 
            #       then operate on them differently, storing only one label
            continue
        break  # Convert only one video.

    return frm_anns

for phase in ['train', 'test']:
    # Get video annotations
    video_anns = pd.read_csv(f"{PATH_TO_CHARADES_ROOT}/Charades_v1_{phase}.csv")

    # Finally, it's time to create the true dataframe we will use.
    print("Creating per-frame annotations (this may take a while)...")
    frame_anns = create_frame_anns(video_anns, PATH_TO_FRAME_DATA)

    print("Converting to dataframe...")
    frame_anns_df = pd.DataFrame(frame_anns, columns=['original_vido_id', 'video_id',
                                                    'frame_id', 'path', 'labels'])
    # print("\nHead of the converted per-frame annotations")
    # print(frame_anns_df.head())

    print("Saving annotations to csv...")
    frame_anns_df.to_csv(f'{PATH_TO_CHARADES_ROOT}/Charades_per-frame_annotations_{phase}.csv', sep=' ', index=False)
