---
tags: HisVis
title: HisVis Model Card - Rotation
author: Melvin Wevers
---

# HisVis Model Card - Rotation Model 

Based on [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993), also see this [post](https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn)

## Model Details 

### Filename
`rotation.pkl`

### Person or organization developing model
The model has been created by Melvin Wevers (University of Amsterdam) in collaboration with the Noord-Hollands Archief. 

### Model Date
March 1, 2022

### Model version 
1.0

### Model Type Information 
This model makes used of a resnet18 model pretrained on imagenet. We use to model to predict whether an image is oriented correctly or incorrectly. 
 

### Parameters:
number of epochs: 50
Early stopping with patience of 5 based on validation loss
learning rate of 0.003
Batch size of 64
Validation size .2
Trained using the Fast.ai one cycle method. 
Augmentations include normalization using imagenet standards, flipping, lighting, rotation, zooming, and scaling. 
Trained on SurfSara computing cluster Lisa
License: CC-NY
Contact details: Melvin Wevers (melvin.wevers@uva.nl)



## Intended Use 
The primary intended user will be the Noord-Hollands Archief, which will use the model to correctly orient image after being cropped from photo negative sheets. Since these images could be in landscape or portrait mode, they need to be oriented correctly. Other users include other heritage institutes. Other users include CV experts and practitioners that would like to apply or improve the models for their collections. 

## Factors
We evaluated using accuracy and the F₁ score per-label. This is quite high, and in cases where the model is incorrect this should not be a big issue. 

## Metrics 

|    |precision|recall|f1-score|support|
|----|---------|------|--------|-------|
|incorrect|0.98|0.99|0.99|5719|
correct|0.98|0.94|0.96|1901|
||||||
|accuracy|||0.98|7620|
|macro avg|0.98|0.97|0.97|7620|
|weighted avg|0.98|0.98|0.98|7620|


## Evaluation Data 
We created a validation set of .2 

## Training Data
The training data was created by transformed a sample of correctly oriented images from the *De Boer Collection*. Using a Python Script, we turned images 90 degrees clockwise, anti-clockwise and 180 degrees. This led to 1901 correctly oriented images and 5719 incorrectly oriented images. 

## Ethical Considerations 
The dataset contains images of funerals and accidents that might cause distress. Moreover, the images of churches and funerals, which are expressions of religious beliefs. Finally, the collection contains images of protests, which are representations of specific political opinions and possibly union membership. Locations can also be identified. However, it should be noted these images are of a historical nature. 

The digitization and preparation of the photographic material was done by a heritage institute, a private company, and a research institute. These were compensated for this using three grants. The annotating was done by crowdworkers, who volunteered but were compensated with physical rewards, such as prints of images in the collection. 

## Caveats and Recommendations
The training data is unevenly balanced. We first experimented with predicted the actual orientation, but since we only need to orient the image to its correct orientation, we decided to go for this simpler task. The training data, however, contains images that are incorrectly oriented in three different manners. 