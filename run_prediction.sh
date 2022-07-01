#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu_short
 
#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

 
#Copy input file to scratch
cp -r $HOME/data/step3 "$TMPDIR"
cp -r $HOME/HisVis2/models "$TMPDIR"

echo "files copied!"
 
#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/HisVis2/src/make_predictions.py --data_path "$TMPDIR"/step3 --model_path "$TMPDIR"/models --output_path "$TMPDIR"/output_dir
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME

 python HisVis2/src/make_predictions.py --data_path data/step3/ --model_path HisVis2/models/ --output_path output_dir/