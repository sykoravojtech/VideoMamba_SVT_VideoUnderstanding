import os
import wget

if __name__ == '__main__':
    print('Download pretrained self-supervised video transfomer weights')
    filename = wget.download("https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth", 
                            out="checkpoints/")
    print('Save weights to checkpoints/')