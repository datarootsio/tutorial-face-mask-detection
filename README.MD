[![maintained by dataroots](https://img.shields.io/badge/maintained%20by-dataroots-%2300b189)](https://dataroots.io)

# Face Mask Detection
The general outline of the model


## Project Intro/Objective
The purpose of this project is ________. (Describe the main goals of the project and potential civic impact. Limit to a short paragraph, 3-6 Sentences)


### Methods Used
* Face Detection
* Image Classification


### Technologies
* Python
* Tensorflow


## Project Description
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

### Data collection
Give more details about datasets and where do we collect

### Data preprocessing
Explain the steps that are done to generate our training set like artificial mask generation

### Face Detection
Give more insights about [retinaface](https://github.com/peteryuX/retinaface-tf2) model. Why we used this? Refer to paper

### Masked or Not Masked Classification
Explain how we fine tuned the mobilenet convnet. Why we used mobilenet?

### Model Performance

#### Overall evaluation

#### Overall evaluation

The following table summarizes the performance of the complete pipeline (i.e. the face detector followed by the classifier). We consider a face to be identified if the face detector produces a bounding box with an intersection over union (IoU) > 0.5 with the ground truth bounding box corresponding to that face.

The two most relevant metrics are the true negative rate (TNR), as it tells us how many of the unmasked faces we detect, and the false negative rate (FNR), as it tells us how many times we incorrectly identify an unmasked face. 

From the table we can see that 128 of the 136 unmasked faces were identified correctly, resulting in **TNR = 94%**. 20 of the 118 masked faces were incorrectly identified as unmasked, resulting in **FNR = 15.7%**. 


|    |      ground truth      |  identified and classified correctly | identified but classified wrongly | not identified by detector |
|----------|:-------------:|------:| ------:|------:|
| masked |  127 |  97 | 20 | 10 |
| unmasked |    136   |   128 | 3 | 5  |

The pipeline also incorrectly identified 8 non-masked faces that didn't match any face in the ground truth.

#### Evaluation of the face detector

As can be seen from the table above, the face detector correctly identifies  **94%** of the images in the test set (i.e. 239 out of 254). Of the 15 faces in the ground truth that it does not detect, 10 are masked and 5 are not masked. Additionally, the face detector outputs 6 bounding boxes for the test set that do not correspond to faces in the ground truth.

#### Evaluation of the mask/no mask classifier

The mask/no mask classifier was evaluated on the faces that were correctly identified by the face detector. The inputs to the classifier were the crops corresponding to faces identified by the face detector. With a classification threshold of 0.5, this results in the statistics listed in the table above. Of course, different thresholds can be used to trade off TNR and FNR. The following plot shows the ROC curve for the mask/no mask classifier.


## Getting Started
Inform how to run this model on dploy.ai

#### Members:

|Name     |  Slack Handle   |
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* Dataroots contact details

TODO:
don't forget to add to instructions: git submodule update --init --recursive
