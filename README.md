# HisVis2
Work for Part 2 of HisVis Project


## Folder Structure

### Data

#### Intermediary
`annotations_step1 to 3` - contains database exports provided by Picturae [TO BE REMOVED]

`metadata-with-connected-files-2022-05-23-csv.csv` metadata linked to images is used for sampling images for step 3. 

#### Processed

##### Annotations
`annotation_data_step1.csv` = processed database export
`annotation_data_step21.csv` = processed database export
`annotation_data_step3.csv` = processed database export

### Notebooks


### Output

### SRC

- `find_correct_rotation.py` - Python script to rotate images until they are correctly oriented. This makes use of the `rotation.pkl` model. This has been trained using the notebook `train_model_rotation.ipynb`



## Training the model 

We can train the model using a Jupyter Notebook or a Python Script that can be called using the `run.sh` bash script. 

