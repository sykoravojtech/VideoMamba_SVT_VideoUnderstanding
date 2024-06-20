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

DUMMY_1 = -1  # placeholder for video_id, unused in the implementation
DUMMY_2 = -1  # placeholder for frame_id, unused in the implementation


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

def get_label(str_labels):
    '''Convert a sequence of (start time, end time, action id) to a sequence of integer labels.'''
    int_labels = []
    if pd.isnull(str_labels):
        return ''
    for item in str_labels.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(str(ACTION_ID_TO_LABEL[action_id]))
    return "," .join(int_labels)

def create_frame_anns(vid_anns, path_to_frame_data):
    """Returns annotations with the desired frame paths,
            given the path to the data and the full video ids"""
    frm_anns = []
    for _, anns_row in tqdm(vid_anns.iterrows(), total=len(vid_anns)):
        vid_id = anns_row['id']
        frm_ids = get_frame_ids(vid_id, path_to_frame_data)
        vid_labels = get_label(anns_row['actions']) if pd.notnull(anns_row['actions']) else ''

        for frm_id in frm_ids:
            frm_path = os.path.join(path_to_frame_data, vid_id, f"{frm_id}.jpg")
            ann_entry = (vid_id, DUMMY_1, DUMMY_2, frm_path, vid_labels)
            frm_anns.append(ann_entry)
    return frm_anns


"""
The following commented lines are strictly intended for demonstration purposes.


# Example for action mapping
print("\nA few mappings from the action to ID supporting dict:")
print(dict(list(ACTION_ID_TO_LABEL.items())[:5]))


print("\nA few action class mappings to integer labels:")
print(action_df.head())

# Example load and print of video anns
print("\nFirst few lines of the video annotations:")
print(video_anns.head())

print("\n A few actions in label format:")
for i in range(5):
    out = get_label(video_anns.head()['actions'].iloc[i])
    print(out if out!="" else "(No action class label found)")


# Example crawl of video ids and print
video_ids = get_video_ids(PATH_TO_FRAME_DATA)
print("\nA few video ids:")
print(video_ids[:10])


# Example crawl of frame paths and print
print("\nA few frame ids for a few videos:")
for video_id in video_ids[:5]:
    frame_ids = get_frame_ids(video_id, PATH_TO_FRAME_DATA)
    print(f"Video ID: {video_id}, Frame IDs: {frame_ids[:5]}")


# Example per-frame path and annotation creation
demo_frame_anns = create_frame_anns(video_anns, video_ids[:5], PATH_TO_FRAME_DATA)
print("\nA few per-frame annotations from head and tail:")
for annotation in demo_frame_anns[:3]:
    print(annotation)
for annotation in demo_frame_anns[-3:]:
    print(annotation)
"""

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
