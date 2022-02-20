import clip
from fastai.vision.all import *

import pandas as pd

import argparse

import glob
import newlinejson
import numpy as np

from helper import *


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)



def get_single_img_features(img_path):
    '''
    extract features from a single image using CLIP
    '''
        
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
  
    with torch.no_grad():
        features = model.encode_image(image)

    return features.cpu().numpy()

def make_avg_prediction(places_model, clip_model, classes, img_path, topk=5):
    # to do implement selection between models and averaging
    output_ = {}
    
    # make places predictions
    label, y, output = places_model.predict(img_path)
    output_places = output.numpy()
    output_places = np.expand_dims(output_places, axis=0)

    
    # make clip predictions
    img_features = get_single_img_features(img_path)
    output_clip = clip_model.predict_proba(img_features)
    
    # soft averaging
    avg = np.average([output_places, output_clip], axis=0)
    
    # select top-k
    topk = topk * -1
    
    best_n = np.argsort(avg, axis=1)[:,topk:]
    probabilities = np.round(np.sort(avg, axis=1)[:,topk:], 2)
    labels = [classes[i] for i in best_n]
    
    for i, _ in enumerate(probabilities[0]):
        output_[labels[0][i]] = probabilities[0][i]
    return output_



def main(input_path, data_path, places_model_path, clip_model_path, output_path):
    current_time = time.strftime("%Y%m%d")

    # setting path variables 
    Path.BASE_PATH = Path(input_path)
    path = Path.BASE_PATH

    data = get_dls(128, 224, path, augment=False)
    classes = data.vocab

    # places model
    places_model = load_learner(places_model_path)

    # clip model
    clip_model = pickle.load(open(clip_model_path, 'rb'))


    # make prediction

    results = list()
    counter = 1

    imgs = glob.glob(data_path + '/**/*.jpg')

    for img in imgs:
        if counter % 100 == 0:
            print(counter)
        serial_number = int(np.floor(counter/500)) + 1
        series = f'random_{serial_number}'
        d = dict()
        filename = os.path.basename(img)[:-4]
        d['filename'] = filename
        d['series'] = series
        d['predictions'] = make_avg_prediction(places_model, clip_model, classes, img)
        
        results.append(d)
        counter += 1

    #save output
    filename = current_time + 'predictions.json'
    with open(os.path.join(output_path, filename), "w") as f:
        newlinejson.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, default='../../../news_nl')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--places_model', type=str, default='../../../news_nl')
    parser.add_argument('--clip_model', type=str, default='../../../news_nl')

    parser.add_argument('--output_path', type=str, default='./output/predictions/')
    args = parser.parse_args()

    if not os.path.exists('./output/predictions'):
        os.makedirs('./output/predictions')
    
    main(args.training_data_path, args.data_path, args.places_model, args.clip_model, args.lr, args.output_path)