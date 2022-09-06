#!/usr/bin/env python
# -*- coding: utf-8 -*-

import clip
from fastai.vision.all import *

import pandas as pd
import random
import argparse

import glob
import newlinejson
import numpy as np

from helper import *


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)



def clip_predict(text_features, dutch_labels, img_path):
    '''
    make predictions using clip. 
    text_features: embeddings of text inputs
    dutch_labels: dutch translations of text inputs
    img_path: location of image

    returns dict with probabilities of predictions per label
 
    '''
    output_ = {}
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    k = len(dutch_labels) # number of labels

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)

    for i, _ in enumerate(values):
        output_[dutch_labels[i]] = np.round(values[i].item(), 2)
    
    return output_


def main(data_path, output_path, n_samples=10000):
    '''
    Here we run through the datapath and take a sample which we input into the 
    CLIP predict function. 
    The text inputs and dutch labels are defined in this function. 
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    current_time = time.strftime("%Y%m%d")
    print(current_time)

    new_labels = ['indoor', 'outdoor']
    dutch_labels = ['binnen', 'buiten']

    text = clip.tokenize(new_labels).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)

    
    # make predictions
    results = list()
    counter = 1
    print(data_path)
    imgs = glob.glob(data_path + '/**/*.jpg', recursive=True)

    # take sample 
    imgs = random.sample(imgs, n_samples)
    print('number of images: {}'.format(len(imgs)))

    for img in imgs:
        if counter % 100 == 0:
            print(counter)
        serial_number = int(np.floor(counter/500)) + 1 # we add a serial number to the sets. 
        series = f'random_in_out_{serial_number}'
        d = dict()
        filename = os.path.basename(img)[:-4]
        d['filename'] = filename
        d['series'] = series
        d['predictions'] = clip_predict(text_features, dutch_labels, img)
        
        results.append(d)
        counter += 1

    #save output
    filename = current_time + 'predictions_in_out.json'
    with open(os.path.join(output_path, filename), "w") as f:
        newlinejson.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../MelvinWevers#9512/VeleHanden')
    parser.add_argument('--output_path', type=str, default='./output/predictions/')
    args = parser.parse_args()

    if not os.path.exists('../output/predictions'):
        os.makedirs('../output/predictions')
    
    main(args.data_path, args.output_path)