#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
 
#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

 
#Copy input file to scratch
cp -r $HOME/data/jp2/VeleHanden\ Fotopersbureau\ De\ Boer/Batch_01 "$TMPDIR"
cp -r $HOME/HisVis2/models "$TMPDIR"
 
#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/HisVis2/src/remove_borders.py "$TMPDIR"/jp2/VeleHanden\ Fotopersbureau\ De\ Boer/Batch_01/**/*.jp2 --resize --crop
python $HOME/HisVis2/src/find_correct_rotation.py  "$TMPDIR"/jp2/VeleHanden\ Fotopersbureau\ De\ Boer/Batch_01/**/*.jp2 --model_path "$TMPDIR"/models/rotation.pkl --output_path "$TMPDIR"/output_dir
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
