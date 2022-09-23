from fastai.vision.all import *
import torch
import clip
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
                                    flip_vert=False,
                                    max_lighting = 0.2,
                                    p_lighting = 0.75,
                                    max_rotate = 10.0, 
                                    max_warp = 0.2,
                                    max_zoom = 1.2, 
                                    min_scale = 0.75), 
                    Normalize.from_stats(*imagenet_stats)] 
        print(batch_tfms)
    else:
        batch_tfms=[*aug_transforms(size=size, min_scale=0.75), Normalize.from_stats(*imagenet_stats)]
        
    dblock = DataBlock(blocks = (ImageBlock, CategoryBlock),
                       get_items=get_image_files, 
                       splitter=RandomSplitter(valid_pct=0.2, seed=42), #change to 0.2
                       get_y=label_func,
                       item_tfms=Resize(460),
                       batch_tfms=batch_tfms)
    return dblock.dataloaders(path, bs=bs)


def new_resnet(pretrained):
    '''
    fix for cnn_learner module using non-standard pre-trained models
    '''

    arch = 'resnet50'

    # (down)load the pre-trained weights
    model_weights = '%s_places365.pth.tar' % arch
    if not os.access(model_weights, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_weights
        os.system('wget ' + weight_url)

    places_res50 = torch.load(model_weights,  
                              map_location=lambda storage, 
                              loc: storage)

    default_res50 = models.resnet50()
    state_dict = places_res50['state_dict'] 
    new_state_dict = OrderedDict()

    for key in state_dict.keys():
        new_state_dict[key[7:]]= state_dict[key]


    default_res50.fc = torch.nn.Linear(2048, 365) # Matching with default res50 dense layer
    default_res50.load_state_dict(new_state_dict)
    return default_res50


    def get_model(model, pretrained=False, progress=True, **kwargs):
        '''
        model: function to load the model, e.g. resnet18
        pretrained, progress: to be passed to the model function
        '''
        m = model(pretrained=pretrained, progress=progress, **kwargs)
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, 2)
        return m
