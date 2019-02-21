# Train fully convolutional neural net for semantic segmntation
# Instructions:
# a) Download pretrained DenseNet model for net initiation from:
#    https://drive.google.com/file/d/1bFdIbIS_2pWd9PQs1x_hYq6Y0BVsL2eI/view?usp=sharing
#    and put in Pretrained_Encoder_Weights folder
# b) Set folder of training images in Train_Image_Dir
# c) Set folder for ground truth labels in Train_Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image in Train_Image_Dir and png ending (the pixel value should be its label)
# d) Set number of classes the net can predict in number in NUM_CLASSES
# e) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir
# f) Run script
# See additional parameters you can playu with in the input parameters section of this file
#...............................Imports..................................................................
import PreProccessing #transfer inputs to pytorch
import Data_Reader  # Read data from files
import NET_FCN # The net Class
import os
import torch
import numpy as np

#...........................................Input parametrs................................................................
Train_Image_Dir="Data_Zoo/Materials_In_Vessels/Train_Images/" # Images and labels for training
Train_Label_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"# Annotetion in png format for train images (assume the name of the images and annotation images are the same (but annotation is always png format))
UseValidationSet=False# do you want to use validation set in training
Valid_Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Validation images that will be used to evaluate training (the ROImap and Labels are in same folder as the training set)
Valid_Label_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/" # Annotetion in png format for validation images (assume the name of the images and annotation images are the same (but annotation is always png format))

Pretrained_Encoder_Weights="Pretrained_Encoder_Weights/densenet_cosine_264_k32.pth" #File were weight for pretrained encoder are saved this can be download from: https://drive.google.com/file/d/1bFdIbIS_2pWd9PQs1x_hYq6Y0BVsL2eI/view?usp=sharing
TrainedModelWeightDir="TrainedModelWeights/" # Folder where trained model weight and information will be stored"
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight

Learning_Rate=1e-5 #Learning rate for Adam Optimizer
MaxPixel=34000# Max pixel image can have (to keep oom out of memory problems) if the image larger it will be resized. Reduce if
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=TrainedModelWeightDir+"ValidationLoss.txt"# Where validation losses will be writen

Batch_Size=1 # Number of images per training iteration (keep small to avoid out of  memory problems)
Weight_Decay=1e-4# Weight for the weight decay loss function
MAX_ITERATION = int(800010) # Max  number of training iteration
NUM_CLASSES = 4#Number of classes the model predict

UpdateEncoderBatchNormStatistics=True

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------

Net=NET_FCN.Net(NumClasses=NUM_CLASSES,PreTrainedModelPath=Pretrained_Encoder_Weights,UpdateEncoderBatchNormStatistics=UpdateEncoderBatchNormStatistics) # Create net and load pretrained encoder path

if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

TrainReader = Data_Reader.Data_Reader(ImageDir=Train_Image_Dir, GTLabelDir=Train_Label_Dir,BatchSize=Batch_Size) #Reader for training data
if UseValidationSet:
    ValidReader = Data_Reader.Data_Reader(ImageDir=Valid_Image_Dir,  GTLabelDir=Valid_Label_Dir,BatchSize=Batch_Size) # Reader for validation data

#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
f = open(TrainLossTxtFile, "w")# Training loss log file
f.write("Iteration\tloss\t Learning Rate="+str(Learning_Rate))
f.close()
if UseValidationSet:
   f = open(ValidLossTxtFile, "w") #Validation  loss log file
   f.write("Iteration\tloss \t  AvgLoss \t Learning Rate=" + str(Learning_Rate))
   f.close()

#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss

for itr in range(1,MAX_ITERATION): # Main training loop

    Images,  GTLabels =TrainReader.ReadAndAugmentNextBatch() # Load  augmeted images and ground true labels for training

    OneHotLabels = PreProccessing.LabelConvert(GTLabels, NUM_CLASSES) #Convert labels map to one hot encoding pytorch

    Prob, Lb=Net.forward(Images) # Run net inference and get prediction
    Net.zero_grad()
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label

    if AVGLoss==-1:  AVGLoss=float(Loss.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLoss=AVGLoss*0.99+0.01*float(Loss.data.cpu().numpy()) # Intiate runing average loss

    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 10000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir)
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        torch.cuda.empty_cache()  #Empty cuda memory to avoid memory leaks
        print("Step "+str(itr)+" Train Loss="+str(float(Loss.data.cpu().numpy()))+" Runnig Average Loss="+str(AVGLoss))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(Loss.data.cpu().numpy()))+"\t"+str(AVGLoss))
            f.close()
#.....................Caclculate Validation Set loss (optional).....................................................................
    if UseValidationSet and itr % 2000 == 0:
        SumLoss=np.float64(0.0)
        NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
        print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
        for i in range(NBatches):# Go over all validation images
            Images, GTLabels= ValidReader.ReadNextBatchClean() # load validation image and ground true labels
            print(Images.shape)
            OneHotLabels = PreProccessing.LabelConvert(GTLabels,NUM_CLASSES)  # Convert labels map to one hot encoding pytorch
            Prob,Lb = Net.forward(Images)  # Run inference and get prediction
            TLoss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and
            SumLoss+=float(TLoss.data.cpu().numpy())
            NBatches+=1
        SumLoss/=NBatches
        print("Validation Loss: "+str(SumLoss))
        with open(ValidLossTxtFile, "a") as f: #Write validation loss to file
            f.write("\n" + str(itr) + "\t" + str(SumLoss))
            f.close()
##################################################################################################################################################

