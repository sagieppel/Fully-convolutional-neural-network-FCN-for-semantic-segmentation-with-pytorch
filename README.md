#  Semantic segmentation with Fully convolutional neural network (FCN) pytorch implementation.

Fully convolutional neural network (FCN) for pixelwise annotation (semantic segmentation) of images implemented on pytorch. 
 

## Details input/output
The input for the net is RGB image (Figure 1 right).
The net produces pixel-wise annotation as a matrix in the size of the image with the value of each pixel corresponding to its class (Figure 1 left).

![](/Figure1.png)
Figure 1) Semantic segmentation of image of liquid in glass vessel with FCN. Red=Empty Vessel, Blue=Liquid Filled Vessel, Grey=Background

## Requirements
This network was run with Python 3.7  [Anaconda](https://www.anaconda.com/download/) package and [Pytorch 1](https://pytorch.org/). The training was done using Nvidia GTX 1080.

## Setup
1) Install [Pytorch](https://pytorch.org/)
2) Download the code from the repository.

## Tutorial

### Instructions for training:
In: TRAIN.py
1) Download pretrained DenseNet model for net initiation from [here](https://drive.google.com/file/d/1bFdIbIS_2pWd9PQs1x_hYq6Y0BVsL2eI/view?usp=sharing]) or [here](https://drive.google.com/file/d/1m1kogoWPkKwBaMkzZxJygHi-mbGbup1Y/view?usp=sharing)
    and put in Pretrained_Encoder_Weights folder
2) Set folder of training images in Train_Image_Dir
3) Set folder for ground truth labels in Train_Label_DIR
   The Label Maps should be saved as png image with same name as the corresponding image in Train_Image_Dir and png ending (the pixel value should be its label)
4) Set number of classes the net can predict in number in NUM_CLASSES
5) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir
6) Run script
See additional parameters you can playu with in the input parameters section of the train.py script

### Semantic segmentation for image using trained net 
In: Inference.py
1) Make sure you you have trained model in Trained_model_path (See Train.py for creating trained model)
2) Set the Image_Dir to the folder where the input image for prediction are located
3) Set number of classes number in NUM_CLASSES
4) Set Output_Dir the folder where you want the output annotated images to be save
5) Run script

### Annotating Video using trained net
In: InferenceVideo.py
1) Make sure you you have trained model in Trained_model_path (See Train.py for creating trained model)
2) Set the InputVid to the input video file
3) Set number of classes  in NUM_CLASSES
4) Set OutputVid to the output video file (with segmentation overlay)
5) Run script


### Evaluating net performance using intersection over union (IOU):
In: Evaluate_Net_IOU.py
1) Make sure you you have trained model in Trained_model_path (See Train.py for training model)
2) Set the Image_Dir to the folder where the input images for prediction are located
3) Set folder for ground truth labels in Label_DIR
    The Label Maps should be saved as png image with same name as the corresponding image and png ending
4) Set number of classes number in NUM_CLASSES
5) Set classes names in Classes
6) Run script

## Net Architecture
The net is based on [fully convolutional neural network for semantic segmentation](https://arxiv.org/pdf/1605.06211.pdf) and composed of [Densenet](https://arxiv.org/pdf/1608.06993.pdf) encoder [PSP](https://arxiv.org/pdf/1612.01105.pdf) itermediate layers  and two [skip connections](https://arxiv.org/pdf/1605.06211.pdf) upsample layers. The net architecture is defined in the NET_FCN.py file. The Densenet encoder is defined in densenet_cosine_264_k32.py.

## Trained models 
1) Trained Model for recognition of fill and emprty region of transperent vessels and glassware ([3 Classes](https://drive.google.com/file/d/1yw7e83ux1F0yrHR1k9PZRQVd37jxUov_/view?usp=sharing)) can be download from [here](https://drive.google.com/file/d/1s4PZXkMn7euMMsxFOIaMKYjOIeSv-ZTJ/view?usp=sharing) 
2) Trained model for recogntion of liquid and solid  materials phases in glassware and transperent vessels ([4 Classes](https://drive.google.com/file/d/1HkwjFU1ffo29oSER3rak5qoLKvpwf9Sn/view?usp=sharing)) can be download from [here](https://drive.google.com/file/d/1vALUddiwnZNpBjum1jCHYkJGYN0eQg7q/view?usp=sharing) 
3) Trained model for recogntion of glassware and transperent vessels (2 Classes) can be download from [here](https://drive.google.com/open?id=13J404gHZy1eSIy3ynCCRrk2M6iLvJuy0)
## Supporting data sets
The net was tested on a dataset of annotated images of materials in glass vessels. 
This dataset can be downloaded from [here](https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing)

MIT Scene Parsing Benchmark with over 20k pixel-wise annotated images can also be used for training and can be download from [here](http://sceneparsing.csail.mit.edu/)
