#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
 
#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

 
#Copy input file to scratch
cp -r $HOME/data/DeBoer_training/set_binnen_buiten "$TMPDIR"

echo "files copied!"
 
#Create output directory on scratch
mkdir "$TMPDIR"/output_dir
 
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.

#python $HOME/HisVis2/src/train_places_model.py --training_data_path "$TMPDIR"/set_binnen_buiten --output_path "$TMPDIR"/output_dir
python $HOME/HisVis2/src/train_clip.py --training_data_path "$TMPDIR"/set_binnen_buiten --output_path "$TMPDIR"/output_dir
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
