import argparse
import glob
import json
import os
import pandas as pd
import shutil 


def process_description(x):
    '''
    turn json columns into usable data. Take only specific value
    '''
    return pd.json_normalize(json.loads(x)).iloc[:, 1:2].values[0][0]


def fix_labels(x):
    '''
    function to fix label names to labels used by training
    '''
    return x.lower().replace(' ','_')

def calculate_disagreement(df):
    disagreement = {}
    for i, group in df.groupby('image_id'):
        if group['annotated_label'].values[0] != group['annotated_label'].values[1]:
            disagreement[group['image_id'].values[0]] = 'disagree'
        else:
            disagreement[group['image_id'].values[0]] = 'agree'
    disagree_df = pd.DataFrame(disagreement, index=['agreement']).T
    df = pd.merge(df, disagree_df, left_on='image_id', right_index=True)
    return df

def calculate_top_n(df):
    in_top_1 = []
    in_top_5 = []

    for i, row in df.iterrows():
        if row['corrected_label'] == row['top_1']:
            in_top_1.append(1)
            in_top_5.append(1)
        elif row['corrected_label'] in row['top_5']:
            in_top_1.append(0)
            in_top_5.append(1)
        else:
            in_top_1.append(0)
            in_top_5.append(0)
    
    df['in_top_1'] = in_top_1
    df['in_top_5'] = in_top_5
    return df



def main(input_path, annotation_step):
    print('Loading data')
    # prepare labels
    results_path = os.path.join(input_path, 'processed', 'resultsRandomBatch10.json')
    predicted_labels = pd.read_json(results_path, lines=True)

    top_1 = []
    top_5 = []
    for k, v in predicted_labels['predictions'].to_dict().items():
        top_1.append(list(v.keys())[-1])
        top_5.append(list(v.keys()))
    predicted_labels['top_1'] = top_1
    predicted_labels['top_5'] = top_5

    annotation_folder = f'annotations_step{annotation_step}'
    # load images file
    images_path = os.path.join(input_path, 'intermediary',  annotation_folder, 'pic_vh_nl_prod_ranh_tagcorrection_deboer_table_images.csv')
    images = pd.read_csv(images_path)

    # load indexeer data
    indexeer_path = os.path.join(input_path, 'intermediary', annotation_folder, 'pic_vh_nl_prod_ranh_tagcorrection_deboer_table_indexeer_data.csv')
    indexeer_data = pd.read_csv(indexeer_path)
    indexeer_data['annotated_label'] = indexeer_data['descriptions'].apply(process_description)

    # load scan data
    scan_path = os.path.join(input_path, 'intermediary', annotation_folder, 'pic_vh_nl_prod_ranh_tagcorrection_deboer_table_scans.csv')
    scans = pd.read_csv(scan_path)

    # load remarks
    opmerkingen_path = os.path.join(input_path, 'intermediary', annotation_folder, 'pic_vh_nl_prod_ranh_tagcorrection_deboer_table_opmerkingen.csv')
    opmerkingen = pd.read_csv(opmerkingen_path)

    # load controle_data
    controle_path = os.path.join(input_path, 'intermediary', annotation_folder, 'pic_vh_nl_prod_ranh_tagcorrection_deboer_table_controle_data.csv')
    controle_data = pd.read_csv(controle_path)
    controle_data['corrected_label'] = controle_data['descriptions'].apply(process_description)

    # merging data
    print('merging data')

    df = pd.merge(predicted_labels[['filename', 'predictions', 'top_1', 'top_5']], images[['title', 'id']], left_on='filename', right_on='title', how='left')
    df.drop('filename', inplace=True, axis=1)
    df.rename(columns = {'id':'image_id'}, inplace = True)

    df = pd.merge(df, scans[['title','id', 'too_difficult', 'unusable']])
    df.rename(columns = {'id':'scan_id'}, inplace = True)

    df = pd.merge(df, indexeer_data[['image_id', 'bewerkt_op', 'gebruiker_id', 'annotated_label']])
    df.rename(columns = {'bewerkt_op':'annotated_on', 'gebruiker_id': 'annotator_id'}, inplace = True)

    df = pd.merge(df, controle_data[['image_id', 'bewerkt_op', 'gebruiker_id', 'corrected_label']], on='image_id')
    df.rename(columns = {'bewerkt_op':'checked_on', 'gebruiker_id': 'validator_id'}, inplace = True)

    df['annotated_label'] = df['annotated_label'].apply(fix_labels)
    df['corrected_label'] = df['corrected_label'].apply(fix_labels)

    df = pd.merge(df, opmerkingen[['scan_id', 'toelichting', 'gebruiker_id']], left_on='scan_id', right_on='scan_id', how='left')

    # calculating disagreement between annotators
    df = calculate_disagreement(df)

    # calculating top_n
    df = calculate_top_n(df)

    # output annotation_data
    print('writing output')
    df.to_csv(os.path.join(input_path, 'processed', f'annotation_data_step_{str(annotation_step)}.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/')
    parser.add_argument('--annotation_step', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists('../data/processed/'):
        os.makedirs('../data/processed')
    
    main(args.input_path, args.annotation_step)
