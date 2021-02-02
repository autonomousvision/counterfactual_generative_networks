#!/usr/bin/env bash

gdown "https://drive.google.com/u/0/uc?id=1VkKexkWh5SeB8fgxAZxLKgmmvDXhVYUy&export=downloadl"
mv u2net.pth imagenet/weights

gdown "https://drive.google.com/u/0/uc?id=12yVFHPUjKmUFGnO2D4xVlTSpF8CUj136&export=download"
mv cgn.pth imagenet/weights

wget "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin"
mv biggan-deep-256-pytorch_model.bin imagenet/weights/biggan256.pth
