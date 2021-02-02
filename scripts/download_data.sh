#!/usr/bin/env bash

# Download Colored MNIST
gdown https://drive.google.com/u/0/uc?export=download&confirm=rHtT&id=1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu
tar -xzvf colored_mnist.tar.gz
mv colored_mnist mnists/data
rm colored_mnist.tar.gz

# Download BG challenge dataset
wget https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz
tar -xzvf backgrounds_challenge_data.tar.gz
mkdir imagenet/data/in9
mv bg_challenge/* imagenet/data/in9
rmdir bg_challenge
rm backgrounds_challenge_data.tar.gz

# Download the Cue Conflict dataset
git clone https://github.com/rgeirhos/texture-vs-shape/
mkdir imagenet/data/cue_conflict
mv texture-vs-shape/stimuli/style-transfer-preprocessed-512/* imagenet/data/cue_conflict
rm -rf texture-vs-shape
