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
    #print(output_)
    return output_


def main(data_path, model_path, output_path):
    current_time = time.strftime("%Y%m%d")
    print(current_time)
    # Path.BASE_PATH = Path(input_path)
    # path = Path.BASE_PATH
    # data = get_dls(128, 224, path, augment=False)
    # classes = data.vocab
    # print(classes)

    # to do load classes from txt file

    classes = np.array(['akker', 'amfitheater', 'aula', 'auto', 'auto_ongeluk', 'bakkerij', 'basketbal_korfbal', 'begraafplaats', 'begrafenis', 'bibliotheek_boekwinkel', 'binnen_zwembad', 'bloemen', 'bloementuin', 'borden_gevelsteen', 'bos_park', 'boten', 'bouwplaats', 'brand', 'brug', 'bruiloft', 'buiten_zwembad', 'bus_truck', 'cafe', 'catwalk', 'circus', 'cricket', 'dansende_mensen', 'demonstratie', 'dieren_overig', 'duinen', 'eend', 'etalage', 'etende_mensen', 'fabriek', 'fietsende_mensen', 'garage_showroom', 'gebouw', 'geestelijken', 'golf', 'groepsportret', 'gymnastiek', 'handbal', 'hardlopen', 'haven', 'herdenking', 'historisch_toneelstuk', 'hockey', 'hond', 'honkbal', 'huisje', 'kade', 'kamperen', 'kantoor', 'kapper', 'kat', 'kerk_binnen', 'kerk_buiten', 'kerstmis', 'keuken', 'klaslokaal', 'koe', 'konijn', 'kunstwerk', 'luchtfoto', 'maquette', 'markt', 'mensen_op_een_boot', 'mensen_op_trap', 'mensenmassa', 'militair', 'motorfiets', 'muziek_optreden', 'ongeluk_brancard', 'ontvangst_afscheid', 'opgraving', 'optocht', 'paard', 'plattegrond', 'portret', 'race', 'roeien', 'schaatsen', 'schaken_dammen', 'scheepswerf', 'sinterklaas', 'slagerij', 'sneeuwlandschap', 'speech', 'speeltuin', 'sport_overig', 'standbeeld', 'straat', 'strand', 'tafel_tennis', 'tentoonstelling', 'terras', 'theater', 'toren', 'tram', 'trein', 'trein_ongeluk', 'trein_station', 'uitreiking_huldiging', 'vechtsport', 'vergaderruimte', 'vijver_plas', 'visserij', 'vlag_hijsen', 'vliegtuig', 'voetbal', 'voetbal_team', 'vogels', 'volleybal', 'waterweg', 'wielrennen', 'windmolen', 'winkel_binnen', 'winkelstraat', 'woonkamer', 'woonwijk', 'zaalvoetbal', 'zeepkistenrace', 'ziekenhuis', 'zwaan'])
    print(classes)

    # setting path variables 

    print(model_path)

    # places model
    places_model = load_learner(os.path.join(model_path, '20220630DeBoerPlaces.pkl'))
    print('model loaded')

    # clip model
    clip_model = pickle.load(open(os.path.join(model_path, '20220630_clip_linear_prob_model.sav'), 'rb'))
    print('model loaded')


    # make prediction

    results = list()
    counter = 1
    print(data_path)
    imgs = glob.glob(data_path + '/**/*.jpg', recursive=True)
    print('number of images: {}'.format(len(imgs)))

    for img in imgs:
        if counter % 10 == 0:
            print(counter)
        serial_number = int(np.floor(counter/500)) + 1
        series = f'batch3_{serial_number}'
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
    #parser.add_argument('--training_data_path', type=str, default='../../data/DeBoer_Step1')
    parser.add_argument('--data_path', default='../../MelvinWevers#9512/VeleHanden')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--output_path', type=str, default='./output/predictions/')
    args = parser.parse_args()

    if not os.path.exists('./output/predictions'):
        os.makedirs('./output/predictions')
    
    main(args.data_path, args.model_path, args.output_path)