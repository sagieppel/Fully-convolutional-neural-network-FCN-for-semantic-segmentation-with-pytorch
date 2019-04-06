# Run prediction on video and genertae pixelwise annotation for every pixels in each video frame
# Output saved as label images, and label overlay on the original image
# 1) Make sure you you have trained model in Trained_model_path (See Train.py for creating trained model)
# 2) Set the InputVid to the input video file
# 3) Set number of classes  in NUM_CLASSES
# 4) Set OutputVid to the output video file (with segmentation overlay)
# 5) Run script
#-----------------------Imports---------------------------------------------------------------------------------------------
import numpy as np
import scipy.misc as misc
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import NET_FCN
import torch
import cv2
#------------------------Input Parameters-------------------------------------------------------
InputVid='Untitled Folder/input.avi'
OuputVid='output.avi'
Trained_model_path="TrainedModelWeights/FillLevelRecognitionNetWeights.torch"# "Path to trained net weights
w=0.8# weight of overlay on image for display
Output_Dir="Output_Prediction/" #Folder where the output prediction will be save
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 3 # Number of classes
Scale=1

#-----------------------------Create net and load weight--------------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=NUM_CLASSES) #Build Net
Net.load_state_dict(torch.load(Trained_model_path)) # Load Traine model
Net.eval()
Net.half()

print("Model weights loaded from: "+Trained_model_path)
#############################Open Video Files###################################################################################################################################################

InStream = cv2.VideoCapture(InputVid)
ret, frame = InStream.read()
# Define the codec and create VideoWriter object
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
FrameSize=(int(frame.shape[1]*Scale),int(frame.shape[0]*Scale))
OutStream = cv2.VideoWriter(OuputVid,fourcc, 20.0,(FrameSize[1],FrameSize[0]))
i=0
#######################################################################################



print("Running Predictions:")
print("Saving output to:" + Output_Dir)
#----------------------Go over video and predict semantic segmentation per frame-------------------------------------------------------------
i = 0

while(InStream.isOpened()):
    ret, frame = InStream.read()
    if ret==True:
        i+=1
        print("Frame "+str(i))
        #frame = cv2.flip(frame,0)

        frame = cv2.resize(frame,FrameSize)
        #frame = np.rot90(frame,3)
        frame = np.expand_dims(frame,axis=0)
        Prob, Pred = Net.forward(frame,EvalMode=True)  # Predict annotation using net
        LabelPred = Pred.data.cpu().numpy()
        OverLay=Overlay.OverLayLabelOnImage(frame[0], LabelPred[0], w)
       # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # write the flipped frame
        #OverLay = cv2.resize(OverLay, FrameSize)
        OutStream.write(OverLay)

        #cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break

        # Release everything if job is finished
InStream.release()
OutStream.release()
