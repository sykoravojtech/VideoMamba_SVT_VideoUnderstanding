import os
import wget

print('Download pretrained self-supervised video transfomer weights')
filename = wget.download("https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth", 
                        out="checkpoints/")

