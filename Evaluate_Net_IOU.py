# Evaluate the perfomance of trained network by evaluating intersection over union (IOU) of the  network predcition
# and ground truth label map of the validation set
# 1) Make sure you you have trained model in Trained_model_path (See Train.py for training model)
# 2) Set the Image_Dir to the folder where the input images for prediction are located
# 3) Set folder for ground truth labels in Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# 4) Set number of classes in NUM_CLASSES
# 5) Set classes names in Classes
# 6) Run script
#######################Imports####################################################################################################################################
import numpy as np
import scipy.misc as misc
import IOU
import Data_Reader
import numpy as np
import torch
import NET_FCN as FCN

#......................Input Parameters...................................................................

Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All//"#Images for evaluation
Label_Dir="Data_Zoo/Materials_In_Vessels/FillLevelLabels"#Ground truth  per pixel annotation for the images in Image dir
Trained_model_path="TrainedModelWeights/FillLevelRecognitionNetWeights.torch"# "Path to trained net weights

NUM_CLASSES = 3 #Number of classes the net predicts
#Classes = ["BackGround", "Empty Vessel","Liquid","Solid"] #names of classe the net predic
#Classes=["Background","Vessel"] #Classes predicted for vessel region prediction
Classes=["BackGround","Empty Vessel region","Filled Vessel region"]#
#Classes=["BackGround","Empty Vessel region","liquid","Solid"]
#Classes=["BackGround","Vessel","Liquid","Liquid Phase two","Suspension", "Emulsion","Foam","Solid","Gel","Powder","Granular","Bulk","Bulk Liquid","Solid Phase two","Vapor"]

 # .........................Build FCN Net...............................................................................................
Net=FCN.Net(NumClasses=NUM_CLASSES) #Build Net
Net.load_state_dict(torch.load(Trained_model_path)) # Load Traine model
print("Model weights loaded from: "+Trained_model_path)
Net.eval()
Net.half()
# -------------------------Data reader for validation image-----------------------------------------------------------------------------------------------------------------------------

ValidReader = Data_Reader.Data_Reader(Image_Dir,GTLabelDir=Label_Dir, BatchSize=1) # build reader that will be used to load images and labels from validation set

#--------------------Sum of intersection from all validation images for all classes and sum of union for all images and all classes----------------------------------------------------------------------------------
Union = np.float64(np.zeros(len(Classes))) #Sum of union
Intersection =  np.float64(np.zeros(len(Classes))) #Sum of Intersection
fim = 0
print("Start Evaluating intersection over union for "+str(ValidReader.NumFiles)+" images")
#===========================GO over all validation images and caclulate IOU============================================================
while (ValidReader.itr<ValidReader.NumFiles):
    print(str(fim*100.0/ValidReader.NumFiles)+"%")
    fim+=1

#.........................................Run Predictin/inference on validation................................................................................
    Images,  GTLabels = ValidReader.ReadNextBatchClean()  # Read images  and ground truth annotation
    Prob,Pred = Net.forward(Images,EvalMode=True)#Predict annotation using trained  net
    PredictedLabels=Pred.cpu().numpy()
#............................Calculate Intersection and union for prediction...............................................................
#   print("-------------------------IOU----------------------------------------")
    CIOU,CU=IOU.GetIOU(PredictedLabels.squeeze(),GTLabels.squeeze(),len(Classes),Classes) #Calculate intersection over union
    Intersection+=CIOU*CU
    Union+=CU

#-----------------------------------------Print Results--------------------------------------------------------------------------------------
print("---------------------------Mean Prediction----------------------------------------")
print("---------------------IOU=Intersection Over Inion----------------------------------")
Sum=0
Num=0
for i in range(len(Classes)):
    if Union[i]>0:
        print(Classes[i]+"\t"+str(Intersection[i]/Union[i]))
        Sum+=Intersection[i]/Union[i]
        Num+=1
print("Mean Class:\t"+str(np.mean(Sum/Num)))


