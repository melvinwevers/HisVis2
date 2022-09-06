#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import shutil

def clean_folder(path, min_files=10):
    '''
    This function removes unneeded folders and labels with images occuring less than 
    min_files. 

    path: directory with training data
    min_files: folder with less than `min_files` will be removed
    '''
    remove_folders =['1_Voorbeeldfotos_bij_labels', 'no_description_found']
    for _ in remove_folders:
        if os.path.exists(os.path.join(path, _)) and os.path.isdir(os.path.join(path, _)):
            shutil.rmtree(os.path.join(path, _))
    folders = ([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]) # get all directories 
    for folder in folders:
        contents = os.listdir(os.path.join(path,folder)) # get list of contents
        if len(contents) < min_files: 
            shutil.rmtree(os.path.join(path,folder))
            print('removed {}'.format(folder)) # this remove folder from training data! 
    with open("all_labels.txt", "w") as output: #fix to write after cleaning
        output.write(str(folders))
    output.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to image directory')
    args = parser.parse_args()
    input_path = args.input_path
    
    clean_folder(input_path)

