from fastai.vision.all import *
import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_single_img_features(img_path):
    '''
    extract features from a single image using CLIP
    '''
        
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
  
    with torch.no_grad():
        features = model.encode_image(image)

    return features.cpu().numpy()


def get_features(model, dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataset):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def label_func(fname):
    '''
    function to grab folder name as label name
    '''
    return fname.parent.name


def get_dls(bs, size, path, augment=True):
    '''
    function to get datablocks
    '''
    if augment:
        batch_tfms=[*aug_transforms(size=size, 
                                    do_flip=True, 
                                    max_lighting = 0.2,
                                    min_lighting = -0.2,
                                    p_lighting = 0.75,
                                    max_rotate = 10.0, 
                                    max_zoom=1.1, 
                                    min_scale=0.75), 
                    Normalize.from_stats(*imagenet_stats)]
    else:
        batch_tfms=[*aug_transforms(size=size, min_scale=0.75), Normalize.from_stats(*imagenet_stats)]
        
    dblock = DataBlock(blocks = (ImageBlock, CategoryBlock),
                       get_items=get_image_files, 
                       splitter=RandomSplitter(seed=42),
                       get_y=label_func,
                       item_tfms=Resize(460),
                       batch_tfms=batch_tfms)
    return dblock.dataloaders(path, bs=128)


# def new_resnet(pretrained):
#     '''
#     fix for cnn_learner module using non-standard pre-trained models
#     '''
#     return default_res50