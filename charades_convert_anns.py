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
ACTION_SAFETY_MARGIN = 0.1  # Cut ACTION_SAFETY_MARGIN seconds
                            #       from the start and end time of action sections.
                            #       This serves to prevent going out of bounds.

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
    frame_ids = [os.path.splitext(f)[0] for f in os.listdir(video_path)
            if os.path.isfile(os.path.join(video_path, f))]
    # Sort frame ids based on the numerical part after the dash
    frame_ids.sort(key=lambda x: int(x.split('-')[-1]))
    return frame_ids

def get_labels(raw_actions):
    """Convert a sequence of (action id, start time, end time) to a sequence of integer labels.
            Returns a 1d list of integers."""
    int_labels = []
    if pd.isnull(raw_actions):
        return ''
    for item in raw_actions.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(ACTION_ID_TO_LABEL[action_id])
    return int_labels

def get_actions(raw_actions):
    """Convert a sequence of (action id, start time, end time) to a nested integer list,
            with frame numbers reconstructed from timestamps. 
            Returns a 2d list, with entries of the form 
            [(int): action label, (int): first action frame, (int): last action frame]"""
    actns = []
    for item in raw_actions.split(';'):
        action = item.split(' ')

        # convert action code to integer action label
        action[0] = ACTION_ID_TO_LABEL[action[0]]

        # convert timecodes in seconds (float) to frame number (int),
        #       as frame number = timecode +- safety margin * FPS
        action[1] = int((float(action[1]) + ACTION_SAFETY_MARGIN) * FPS)
        action[2] = int((float(action[1]) - ACTION_SAFETY_MARGIN) * FPS)

        actns.append(action)
    return actns


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
            if pd.notnull(anns_row['actions']):
                video_actions = get_actions(anns_row['actions'])
                chunk_iterator = 0
                for video_action in video_actions:
                    vid_id = vid_id + str(chunk_iterator)  # use different video id for each action
                    chunk_iterator += 1
                    vid_action_label = video_action[0]
                    # since frame ids are now sorted, we can iterate numerically
                    for i in range(video_action[1], video_action[2]):  # from action start to end
                        frm_path = os.path.join(path_to_frame_data, vid_id, f"{frm_ids[i]}.jpg")
                        ann_entry = (vid_id, DUMMY_1, DUMMY_2, frm_path, vid_action_label)
                        frm_anns.append(ann_entry)
            else:
                continue  # Skip videos with no action labels
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
