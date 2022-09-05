from fastai.vision.all import *
import argparse

import os
import clip
import torch

import pandas as pd

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from tqdm import tqdm

import time

from helper import *

# Finetuning CLIP needs more iterations if there's too much data. Perhaps take a sample?

def main(training_path, param_sweep, output_path):

    current_time = time.strftime("%Y%m%d")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # setting path variables 
    Path.BASE_PATH = Path(training_path)
    #Path.BASE_PATH.ls()
    path = Path.BASE_PATH

    data = get_dls(128, 224, path, augment=False)

    # Calculate the image features
    train_features, train_labels = get_features(model, data.train)
    test_features, test_labels = get_features(model, data.valid)

    # training logistic regression classifier
    C = 0.65

    if param_sweep:
        C_param_range = [0.5, 0.55, 0.575, 0.6, 0.65, 0.7]

        acc_Table = pd.DataFrame(columns = ['C_parameter','Accuracy'])
        acc_Table['C_parameter'] = C_param_range

        j = 0 
        for i in C_param_range:
            print(i)
            classifier = LogisticRegression(random_state=0, C = i, max_iter=3000, n_jobs=-1)
            classifier.fit(train_features, train_labels)
        
            predictions = classifier.predict(test_features)
            
            acc_Table.iloc[j, 1] = np.mean((test_labels == predictions).astype(np.float)) * 100.
            
            j += 1
        
        C = acc_Table[acc_Table['Accuracy'] == acc_Table['Accuracy'].max()]['C_parameter'].values


    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=C, max_iter=10000, verbose=1, n_jobs=-1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

    test_labels_ = [data.vocab[x] for x in test_labels]
    predictions_ = [data.vocab[x] for x in predictions]

    print(classification_report(test_labels_, predictions_))

    report = classification_report(test_labels_, predictions_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    filename = current_time + '_classification_reportCLIP.csv'
    report_df.to_csv(os.path.join(output_path, filename))

    ## Calculate top-5 accuracy
    correct = []
    pred_5 = [] 

    top5 = 0.0 

    probs = classifier.predict_proba(test_features)
    best_n = np.argsort(probs, axis=1)[:,-5:]

    for i, preds in enumerate(best_n):
        if test_labels[i] in best_n[i]:
            top5 += 1.0

    print("top5 acc", top5/len(best_n)) 


    # Save Model
    filename = current_time + '_clip_linear_prob_model.sav'
    pickle.dump(classifier, open(os.path.join(output_path, filename), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, default='/DeBoer_training/final_set')
    parser.add_argument('--param_sweep', help='sweep parameters', action='store_true', default=False)
    ## Whether to use parameter sweep over logistic classifier setting.
    parser.add_argument('--output_path', type=str, default='../output/models')
    args = parser.parse_args()

    if not os.path.exists('../output/models'):
        os.makedirs('../output/models')
    
    main(args.training_data_path, args.param_sweep, args.output_path)
