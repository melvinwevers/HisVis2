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



def clip_predict(text_features, dutch_labels, img_path, k=2):
    output_ = {}
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)

    for i, _ in enumerate(values):
        output_[dutch_labels[i]] = np.round(values[i].item(), 2)
    
    return output_





def main(data_path, output_path, k=10000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    current_time = time.strftime("%Y%m%d")
    print(current_time)
    # Path.BASE_PATH = Path(input_path)
    # classes = Path.BASE_PATH.ls()
    # # data = get_dls(128, 224, path)
    # # classes = data.vocab
    # print(classes)

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
    imgs = random.sample(imgs, k)
    print('number of images: {}'.format(len(imgs)))

    for img in imgs:
        if counter % 100 == 0:
            print(counter)
        serial_number = int(np.floor(counter/500)) + 1
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
    #parser.add_argument('--training_data_path', type=str, default='../../data/DeBoer_Step1')
    parser.add_argument('--data_path', default='../../MelvinWevers#9512/VeleHanden')
    #parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--output_path', type=str, default='./output/predictions/')
    args = parser.parse_args()

    if not os.path.exists('./output/predictions'):
        os.makedirs('./output/predictions')
    
    main(args.data_path, args.output_path)