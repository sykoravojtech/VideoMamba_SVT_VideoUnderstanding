import os
import sys
import glob
import tarfile # for extracting tar files
import zipfile # for unzipping files
import rarfile
import requests # for downloading files
import shutil # for moving files

from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
from tqdm import tqdm # for progress bar 

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True) # progress bar
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong")
    print(f'Download complete: {local_filename}')

def unzip_file(zip_file_path, extract_to_path, compressed_type='zip'):
    print(f"Unzipping {zip_file_path} to {extract_to_path}")
    if compressed_type == 'zip':
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
    elif compressed_type == 'rar':
        with rarfile.RarFile(zip_file_path) as rf:
            # Extract all files to the output directory
            rf.extractall(path=extract_to_path)
    else:
        raise ValueError(f'No such compressed type: {compressed_type}')
    print(f'Extraction complete: Files extracted to {extract_to_path}')

def move_files_to_target(src_folder, target_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            src_file_path = os.path.join(root, file)
            shutil.move(src_file_path, target_folder)
        # Remove empty directories
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))

####################################################################

def get_args() -> ArgumentParser:
    """
    Get arguments from the command line
    -ucf, -charades defaults to False but adding it to command line makes it True
    """
    parser = ArgumentParser(description="Download Datasets by adding arguments to the command line")

    parser.add_argument('-ucf', action='store_true', help='Set this flag to True if -ucf is present')
    parser.add_argument('-charades', action='store_true', help='Set this flag to True if -charades is present')
    parser.add_argument('-hmdb51', action='store_true', help='Set this flag to True if -hmdb51 is present')
    
    # Check if no arguments were provided
    if len(sys.argv) == 1:
        print("!!! No arguments provided !!!")
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def download_ucf101() -> None:
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")

    with tarfile.open(file_path) as t:
        t.extractall("data/raw/")


def download_url_dataset(url: str, final_dir: str = "data/raw", 
                        remove_zip: bool = True, compressed_type='zip'):
    # download the zip file
    zip_path = os.path.join(final_dir, os.path.basename(url))
    if not os.path.exists(zip_path):
        download_file(url, zip_path)

    # extract the zip file
    os.makedirs(final_dir, exist_ok=True)
    unzip_file(zip_path, final_dir, compressed_type=compressed_type)

    # Remove the leftover zip file
    if remove_zip:
        os.remove(zip_path)
        print(f'Removed zip file: {zip_path}')
        

if __name__ == "__main__":
    args = get_args()

    if args.ucf:
        if os.path.exists("data/raw/UCF101_subset"):
            print("UCF101 dataset already downloaded.")
        else:
            print("Setting up UCF101 dataset...")
            download_ucf101()
            print("UCF101 dataset setup complete.")
    
    if args.charades:
        if os.path.exists("data/raw/Charades/videos") and os.listdir("data/raw/Charades/videos"): 
            print("Charades dataset already downloaded.")
        else:
            print("...Obtaining Charades annotations...")
            download_url_dataset(
                url = 'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip',
                final_dir = "data/raw",
                remove_zip = True)

            # If you have the zip file of the videos downloaded put it in "data/raw/Charades"
            # That will make it not download again
            print("...Obtaining Charades videos...")
            download_url_dataset(
                url = 'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip',
                final_dir = "data/raw/Charades",
                remove_zip = False)

            os.rename("data/raw/Charades/Charades_v1_480", "data/raw/Charades/videos")
            print("...Charades dataset setup complete...")
        
    if args.hmdb51:
        if os.path.exists("data/raw/HMDB51/HMDB51_videos") and os.listdir("data/raw/HMDB51/HMDB51_videos"): 
            print("HMDB51 dataset already downloaded.")
        else:
            print("...Obtaining HMDB51 videos...")
            os.makedirs("data/raw/HMDB51", exist_ok=True)
            download_url_dataset(
                url = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar',
                final_dir = "data/raw/HMDB51",
                remove_zip = False,
                compressed_type='rar')

            os.makedirs("data/raw/HMDB51/HMDB51_videos")
            for path in tqdm(glob.glob("data/raw/HMDB51/*.rar")):
                if "hmdb51_org.rar" in path:
                    continue

                with rarfile.RarFile(path) as rf:
                    rf.extractall(path="data/raw/HMDB51/HMDB51_videos")

            # download fold split metadata
            download_url_dataset(
                url = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar',
                final_dir = "data/raw/HMDB51",
                remove_zip = True,
                compressed_type='rar')

            # remove zip file
            os.system("rm data/raw/HMDB51/*.rar")