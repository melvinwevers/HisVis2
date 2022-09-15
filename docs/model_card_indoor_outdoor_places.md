---
tags: HisVis
title: HisVis Model Card - Inside Outside Detection Places-365
author: Melvin Wevers
---

# HisVis Model Card - Inside Outside Detection Places-365 

Based on [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993), also see this [post](https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn)

## Model Details 

### Person or organization developing model
The model has been created by Melvin Wevers (University of Amsterdam) in collaboration with the Noord-Hollands Archief. 

### Model Date
2022-09-14

### Model version 
1.0

### Model Type Information 
The model is based on the pre-trained Places 365 model. We apply transfer learning using new images and classes to fine-tune the model.  

### Parameters:
number of epochs: 50
Early stopping with patience of 5 based on validation loss
learning rate of 0.0003
Batch size of 64
Validation size .2
Trained using the Fast.ai one cycle method. 
Augmentations include normalization using imagenet standards, flipping, lighting, rotation, zooming, and scaling. 
No limitations were set on the minimum number of images per class. 
Trained on SurfSara Computing Cluster Lisa
License: CC-NY
Contact details: Melvin Wevers (melvin.wevers@uva.nl)


## Intended Use 
The primary intended user will be the Noord-Hollands Archief, which will use the model to enrich their collections. Other users include other heritage institutes. Other users include CV experts and practitioners that would like to apply or improve the models for their collections. 

## Factors
We evaluated using accuracy and the F₁ score per-label. This is quite high, and in cases where the model is incorrect this should not be a big issue. In these cases, it often even difficult for a human observer to discern whether a picture is taken in an indoor or outdoor setting
Metrics. 


|precision|recall|f1-score|support|
|---------|------|--------|-------|
binnen|0.91|0.95|0.93|757.0
buiten|0.96|0.92|0.94|902.0
accuracy||0.93|1659|
macro avg|0.93|0.93|0.93|1659.0
weighted avg|0.93|0.93|0.93|1659.0

## Evaluation Data 
We created a validation set of .2 

## Training Data
See: From De Boer collection, the crowd annotated images as either inside (n=3,799) and outside (n = 4,498). 

## Ethical Considerations 
The dataset contains images of funerals and accidents that might cause distress. Moreover, the images of churches and funerals, which are expressions of religious beliefs. Finally, the collection contains images of protests, which are representations of specific political opinions and possibly union membership. Locations can also be identified. However, it should be noted these images are of a historical nature. 

The digitization and preparation of the photographic material was done by a heritage institute, a private company, and a research institute. These were compensated for this using three grants. The annotating was done by crowdworkers, who volunteered but were compensated with physical rewards, such as prints of images in the collection. 

## Caveats and Recommendations
The image are to a large extent black and white. We haven't checked how well the model performs on color images. 