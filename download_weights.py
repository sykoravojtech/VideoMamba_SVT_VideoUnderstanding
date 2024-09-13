import os
import wget
import gdown

if __name__ == '__main__':
    print('Download pretrained self-supervised video transfomer weights')
    if not os.path.exists('checkpoints/kinetics400_vitb_ssl.pth'):
        filename = wget.download("https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth", 
                                out="checkpoints/")
        print('Save weights to checkpoints/')
    else:
        print('File kinetics400_vitb_ssl.pth already exists. Skipping download.')


    print('Download the pretrained classification and captioning weights of the project')
    # The Google Drive file ID
    file_id = '1vCcsXaN4E2NuWrUkCYnpBgbUoXSjTKxb'  # Example: 1A2B3C4D5E6F7G8H9I0J

    # Generate the download URL
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Download the file
    output = 'checkpoints/trained_weights.zip'  # Name to save the file as
    if not os.path.exists('checkpoints/trained_weights.zip'):
        gdown.download(download_url, output, quiet=False)
        print(f"Downloaded file saved as: {output}")
    else:
        print('File trained_weights.zip already exists. Skipping download.')
    
    os.system('unzip checkpoints/trained_weights.zip -d checkpoints/')