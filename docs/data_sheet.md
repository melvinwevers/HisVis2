---
title: De Boer Press Photography Data Sheet
authors: Melvin Wevers, Nico Vriend, Alexandra Barancová, Jan Kruidhof, Lars Vereecken
tags: HisVis 
---


# De Boer Press Photography Data Sheet
Melvin Wevers, Nico Vriend, Alexandra Barancová, Jan Kruidhof, Lars Vereecken


Following [Gebru et al. 2021.](https://dl.acm.org/doi/10.1145/3458723), we constructed a datasheet for the dataset used in the HisVis project. Datasheets encourage creators of the data to reflect on the "process of creating, distributing, and maintaining a dataset" as well as on the implications of its use. For users of the data, this datasheet assists in determining whether required information is provided to make proper and informed use of the data and its resulting models. For more on the model, see the HisVis Model Card and more on the labels used, see the HisVis Label Sheet.

## Motivation
### For what purpose was the dataset created? 
This dataset was created as part of the projects HisVis and _Fotografisch Geheugen_. These projects examined to what extent Computer Vision, and more specifically, scene detection could be applied to a collection of historical press photographs. The key aspect of scene recognition is to identify the place in which the objects seat. 

The specific aim of the enrichments provided by scene detection was to benefit users of the archive and cultural historians studying historical photographs.

### Who created the dataset and on behalf of which entity?
The dataset was created by the Noord-Hollands Archief. Digitization was done by LOTS Imaging, while the database was created by Picturae. During digitization, photo negative sheets were scanned. Melvin Wevers (University of Amsterdam) removed the borders and correctly oriented the pictures. The scripts for these operations can be found on [GitHub](https://github.com/melvinwevers/HisVis2/).

### Who funded the creation of the dataset?
The HisVis project was funded by the Dutch Research Council’s Knowledge and Innovation Mapping grant ([NWO KIEM](https://www.nwo.nl/en/projects/ki18034)). The _Fotografisch Geheugen_ project was funded by grants from the Prins Bernhard Cultuurfonds and the Mondriaan Fonds. The digitization of the collection is funded by the Noord-Hollands Archief (Haarlem, The Netherlands).

## Composition
### What do the instances that comprise the dataset represent?
The dataset consists of historical press photographs from the collection _Fotopersbureau De Boer_ (1945-2004). These mostly include pictures taken in the region around Haarlem and  Kennemerland, but it also includes pictures taken nationally. Since these are press photographs they often include registrations of specific events, such as openings, sport events, theater plays, etc. For an overview of the types of scenes in the images see the HisVis Label Sheet. 

### How many instances are there in total?
52,160 randomly selected images from the collection Fotopersbureau De Boer (1945-2004), Noord-Hollands Archief. 

We also selected 8,297 images that were annotated for being taken indoor or outdoor. 

### Does the dataset contain all possible instances or is it a sample of instances from a larger set? 
This dataset is a sample of the larger digitization collection of _Fotopersbureau De Boer_ press photographs. The collection as a whole consists of around two million 135mm film photos, 65,000 roll film, 40,000 sheet film, and 6,000 photographic plates.


### What data does each instance consist of? 
The data consists of images that were cropped out of the photo negative sheets, resized and correctly rotated. The latter was done using a neural network specifically trained for this purpose. 

- [ ] add error margin for rotation. @melvinw 

### Is there a label or target associated with each instance?
Each instance is associated with a label, however we have included the category 'no_description_found' and a few miscellaneous categories, for example 'animals_misc', which contains images of animals for which a more specific label was missing. More details on the labels can be found in the HisVis Label Sheet. 

### Are there recommended data splits? 
The data is split randomly using a twenty percent validation split, meaning that 80 percent of each label is used as training data. 

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?
The dataset is self-contained. It also links to the original photo negative sheets, which are stored by Picturae and presented on the Noord-Hollands Archief's [website](https://noord-hollandsarchief.nl/beeldbankdeboer).

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals’ non-public communications)?
No

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
The dataset contains images of funerals and accidents, as well as images of Sinterklaas, which contain images of people in black face that might cause discomfort and distress. 


### Does the dataset relate to people?
People are featured in the images. The agency Fotopersbureau de Boer sold its photographs to local Haarlem newspapers, as well as major Dutch newspapers and magazines; this meant that the images were used to report on events relating to real people, both regionally and nationally. These people ranged from celebrities to unknown people, often living in the specific regional community. 

### Does the dataset identify any subpopulations (e.g., by age, gender)?
No.


### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?
There are people that are celebrities, who might be recognized. The metadata kept by the press photo agency often contains information on who is represented in the images. For less well-known people, it is possible to indirectly trace the same person in multiple images, providing information on them. 

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?
The data contains images of churches and funerals, which are expressions of religious beliefs. Moreover, the collection contains images of protests, which are representations of specific political opinions and possibly union membership. Locations can also be identified. However, it should be noted these images are of a historical nature. 

## Collection Process
### How was the data associated with each instance acquired?
The image data is directly observable. The labels were produced in an interaction between a model and crowd annotators. Predictions were generated using a pre-trained Places-365 model in combination with a classifier trained on top of CLIP embeddings. Two validators per image validated the predictions. These annotators were presented with the top-5 predictions, but could also select the other labels from a drop-down menu, or choose to skip the image, indicating that the correct label was not available, or that the image was unusable. For the first and second batch of images, both annotations were also checked by a validator. These included annotators with experience. After the second batch, only images with different annotations were presented to the validator. After these corrections, this data was used to update the model and to generate predictions on a new batch of images. We went through this process three times. 

[Include image?]

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?
The photo negatives were digitized at high resolution, using a new digitisation method. During digitisation, photo negative sheets are placed on a table that can be moved along its x and y-axis. A camera above the table takes a picture each time specific coordinates are reached. This [method](https://noord-hollandsarchief.nl/ontdekken/nhalab/835-digitalisering-de-boer) can scan up to fifteen thousand negatives per day.

The labels were captured using the crowd annotation platform [Vele Handen](https://velehanden.nl/). Batches of training data were selected using random samples in combination with text searches of the metadata to find training data for labels with few training images. 

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
This dataset was created in three stages. During the first stage, we randomly selected 10 thousand images on which we ran predictions using a pre-trained model. The labels associated with these images as well as the images themselves were then validated by the crowd using the annotation platform Vele Handen. Some images changed labels, lost labels, or were marked as unusable images. During the second stage, we increased the random sample to 20 thousand images. After the crowd-sourced validation, we estimated the twenty-five labels with the lowest accuracy, images for which the accuracy decreased between step 1 and 2, and labels with fewer than fifty images. Using the collection's metadata, we gathered images for these labels.

[Here](https://github.com/melvinwevers/HisVis2/blob/main/notebooks/5.analyze_annotations.ipynb) you can find a Jupyter Notebook detailing this process. The sample is selected in the first two steps deterministically using the Python sampling function. For the third step this was augmented using specific text queries. This has led to 17.5 thousand new images which we supplemented with another 10 thousand random images. During the final crowd annotation process, only 16.5 thousand images were corrected because we ran out of time. In total, just over 52 thousand images were labeled. For approximately 1,200 images, annotators were unable to assign labels. 

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
The digitization and preparation of the photographic material was done by a heritage institute (Noord-Hollands Archief), a private company (LOTS imagining, Picturae), and a research institute (University of Amsterdam). These were compensated for this using three grants. The annotating was done by 340 crowdworkers, who volunteered but were compensated with physical rewards, such as prints of images in the collection. 

### Over what timeframe was the data collected?
The images of the photographers of _Fotopersbureau De Boer_ cover sixty years and were made between 1945 and 2004. 

The labels were collected between November 2021 and August 2022 
[als het alleen om VeleHanden gaat, eigenlijk al vanaf pilot in [voorjaar 2020?]

### Were any ethical review processes conducted (e.g., by an institutional review board)?
No

### Does the dataset relate to people?
Yes. Also see Does the dataset relate to people?

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
Noord-Hollands Archief obtained the data and its copyright from Fotopersbureau De Boer press agency. 

### Were the individuals in question notified about the data collection?
The previous copyright holders were informed. 

### Did the individuals in question consent to the collection and use of their data?
No. The collection of Fotopersbureau De Boer was formed in the past (1945-2004), which complicates any option to ask consent of portrayed individuals.
 
### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?
Individuals can object to publication of specific images. This can be done directly at the image in question online. Information on this option is provided online at record level, at the homepage of the online image collection of Fotopersbureau De Boer, and on the website of the Noord-Hollands Achief.

https://noord-hollandsarchief.nl/beeldbankdeboer 
https://noord-hollandsarchief.nl/over-ons/organisatie/privacyverklaring 
 
Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis)been conducted?

Noord-Hollands Archief carried out a Privacy Impact Assessment (PIA) on the online publication of the collection of Fotopersbureau De Boer.

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?
The images were cut out from the photo negative sheets and correctly rotated. Each image was labeled to indicate the depicted ‘scene’. 

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
Yes. This data is stored by Picturae. 

### Is the software used to preprocess/clean/label the instances available?
Yes. 
- [GitHub](https://github.com/melvinwevers/HisVis2)
- [Vele Handen](https://velehanden.nl/)

## Uses

### Has the dataset been used for any tasks already? 
The dataset has been used to create an image classification model for scene detection. This model is used to enrich unlabelled parts of the De Boer collection with information on the scenes. 

### Is there a repository that links to any or all papers or systems that use the dataset?


### What (other) tasks could the dataset be used for? 
The dataset could for example also be used to study stylistic changes in photography, linking historical images to contemporary geolocations, historical studies into protest movements. 

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? 
The category Sinterklaas contains depictions of Black Pete. These images can be offensive and could also skew the classification of images with black people. Future users could choose to leave this category containing stereotypical imagery out of their training. Moreover, the number of images for each label is unevenly distributed, which impacts the accuracy of the model. Future uses might want to include more images, especially for classes with few images. 

Moreover, the dataset has been developed for scene detection in historical press photographs, but limited to the period 1945-2004, predominantly in the context of Noord-Holland. This might impact the generalizability of the model. 

### Are there tasks for which the dataset should not be used? If so, please provide a description
The dataset should not be used for face detection and tracing individuals across images. 

## Distribution 

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 
The dataset will be placed on Zenodo

### How will the dataset be distributed (e.g., tarball on website, API, GitHub)? 
The dataset will be distributed using Zenodo. The images will be in the form of a tarball with metadata available as well as script detailing how the data was created. 

### Does the dataset have a digital object identifier (DOI)? 
Yes (although not yet)

### When will the dataset be distributed? 
September 2022

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? 
The Noord-Hollands Archief holds the rights of all images from the collection Fotopersbureau De Boer included in the dataset, and has made the entire collection available for use with a Creative Commons Zero public domain license (CC0). This means you can download and use all images from Beeldbank Fotopersbureau De Boer and this dataset for free. However, when using an image or this dataset, source citation is much appreciated.

Creative Commons — CC0 1.0 Universal 
https://noord-hollandsarchief.nl/beeldbankdeboer 
https://noord-hollandsarchief.nl/collecties/beeld/collectie-fotopersbureau-de-boer/overbeeldbankdeboer-en 

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances? 
No. If you feel that the publication of certain images should only have been done with your approval, please contact us through info@noord-hollandsarchief.nl.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? 
No


## Maintenance

### Who is supporting/hosting/maintaining the dataset? 
The hosting is supported by Zenodo. The raw data is hosted and maintained by Noord-Hollands Archief

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)? 

melvin.wevers@uva.nl
nico.vriend@noord-hollandsarchief.nl
info@noord-hollandsarchief.nl 

### Is there an erratum? 
If so, please provide a link or other access point. 

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? 
Yes. The new dataset and resulting models will be uploaded to Zenodo. These updates will be communicated through the change log. 


