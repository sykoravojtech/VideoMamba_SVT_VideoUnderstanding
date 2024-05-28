import os
import sys
import tarfile
import zipfile
import requests
import shutil

from argparse import ArgumentParser
from huggingface_hub import hf_hub_download


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f'Download complete: {local_filename}')

def unzip_file(zip_file_path, extract_to_path):
    print(f"Unzipping {zip_file_path} to {extract_to_path}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
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


def download_url_dataset(url: str, final_dir: str = "data/raw", remove_zip: bool = True):
    # download the zip file
    zip_path = os.path.join(final_dir, os.path.basename(url))
    if not os.path.exists(zip_path):
        download_file(url, zip_path)

    # extract the zip file
    os.makedirs(final_dir, exist_ok=True)
    unzip_file(zip_path, final_dir)

    # Remove the leftover zip file
    if remove_zip:
        os.remove(zip_path)
        print(f'Removed zip file: {zip_path}')
        

if __name__ == "__main__":
    args = get_args()

    if args.ucf:
        print("Setting up UCF101 dataset...")
        download_ucf101()
    
    if args.charades:
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
        