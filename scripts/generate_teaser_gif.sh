#!/usr/bin/env bash
# Sample an image of class $1, than interpolate to shape $2,
# then background $3, then shape $4, and finally back to $1
echo "Transformations: $1 -> $2 (Text) -> $3 (BG) -> $4 (Shape) -> $1"

# Generate the interpolations
python imagenet/generate_data.py --weights_path imagenet/weights/cgn.pth --trunc 0.4 --run_name "$1_interp1" --n_data 1 --mode fixed_classes --save_single --midpoints 50 --interp text --classes $1 $1 $1 --interp_cls $2 --save_noise
python imagenet/generate_data.py --weights_path imagenet/weights/cgn.pth --trunc 0.4 --run_name "$1_interp2" --n_data 1 --mode fixed_classes --save_single --midpoints 50 --interp bg --classes $1 $2 $1 --interp_cls $3
python imagenet/generate_data.py --weights_path imagenet/weights/cgn.pth --trunc 0.4 --run_name "$1_interp3" --n_data 1 --mode fixed_classes --save_single --midpoints 50 --interp shape --classes $1 $2 $3 --interp_cls $4
python imagenet/generate_data.py --weights_path imagenet/weights/cgn.pth --trunc 0.4 --run_name "$1_interp4" --n_data 1 --mode fixed_classes --save_single --midpoints 50 --interp all --classes $4 $2 $3 --interp_cls $1

# collect the ims in a new directory
rm imagenet/data/u_fixed.pth
find_newest_subdir(){
  echo $(ls -td imagenet/data/* | head -4)
}
newest_dirs=$(find_newest_subdir)

sum_name="transform_$1_T$2_BG$3_S$4"
mkdir "imagenet/data/$sum_name/"

echo $newest_dirs
for d in $newest_dirs
do
  echo "Moving ims from $d"
  cp "$d/ims/"* "imagenet/data/$sum_name/"
  rm -rf "$d"
done

# produce the gif
ffmpeg -framerate 25 -pattern_type glob -i 'imagenet/data/'$sum_name'/*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p $sum_name.mp4
