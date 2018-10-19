
import scipy.misc as misc
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import densenet_cosine_264_k32
class Net(nn.Module):# FCN Net class for semantic segmentation init generate net layers and forward run the inference
        def __init__(self,NumClasses,PreTrainedModelPath="",UseGPU=True,UpdateEncoderBatchNormStatistics=False): # prepare net layers and load Load pretrained encoder weights
            super(Net, self).__init__()
            self.UseGPU = UseGPU
#---------------Load Densenet pretrained encoder----------------------------------------------------------
            self.Encoder = densenet_cosine_264_k32.densenet_cosine_264_k32
            if not PreTrainedModelPath=="":
                self.Encoder.load_state_dict(torch.load(PreTrainedModelPath)) #load densenet encoder pretrained weights
                print ("Dense net encoder weights loaded")
            if not UpdateEncoderBatchNormStatistics:
                self.Encoder.eval()# Weather or not to update batch statistics of encoder during training


            self.SkipConnectionLayers=[2,12,28,96]#,147]
#----------------PSP Layer (resize feature maps to  several scales, apply convolution resize back and concat) -------------------------------------------------------------------------
            self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8] # Resize scales

            self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2688, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
                # nn.BatchNorm2d(1024)))
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
#------------------Skip conncetion pass layers from the encoder to layer from the decoder/upsampler after convolution-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1152, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
# ------------------Skip squeeze concat of upsample+skip conncecion-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256+512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ))


#----------------Final prediction layer predict class per region/pixel------------------------------------------------------------------------------------------
            self.FinalPrdiction=nn.Conv2d(256, NumClasses, stride=1, kernel_size=3, padding=1, bias=False)
            if self.UseGPU==True:
                self=self.cuda()
##########################################################################################################################################################
        def forward(self,Images,EvalMode=False):

#----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False,volatile=EvalMode).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)

                if self.UseGPU == True:
                    InpImages=InpImages.cuda()

                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------Run Encoder------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
                for i in range(147): # run all layers of Encoder
                    x=self.Encoder[i](x)
                    if i in self.SkipConnectionLayers: # save output of specific layers used for skip conncections
                         SkipConFeatures.append(x)
                         #print("skip")
#------------------Run psp  decoder Layers----------------------------------------------------------------------------------------------
                PSPSize=(x.shape[2],x.shape[3]) # Size of the original features map

                PSPFeatures=[] # Results of various of scaled procceessing
                for i,Layer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
                      NewSize=np.ceil(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
                      y = F.upsample(x, tuple(NewSize), mode='bilinear')
                      #y = F.upsample(x, torch.from_numpy(NewSize), mode='bilinear')
                      y = Layer(y)
                      y = F.upsample(y, PSPSize, mode='bilinear')
                #      if np.min(PSPSize*self.ScaleRates[i])<0.4: y*=0
                      PSPFeatures.append(y)
                x=torch.cat(PSPFeatures,dim=1)
                x=self.PSPSqueeze(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=F.upsample(x,size=sp,mode='bilinear') #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)
#---------------------------------Final prediction-------------------------------------------------------------------------------
                x = self.FinalPrdiction(x) # Make prediction per pixel
                x = F.upsample(x,size=InpImages.shape[2:4],mode='bilinear') # Resize to original image size
                Prob=F.softmax(x,dim=1) # Calculate class probability per pixel
                tt,Labels=x.max(1) # Find label per pixel
                return Prob,Labels
###################################################################################################################################

# nt=Net(12).cuda()
# #torch.save(nt,"tt.torch")
# #nt.save_state_dict("aa.pth")
# inp=np.ones((1,3,1000,1000)).astype(np.float32)
# inp=torch.autograd.Variable(torch.from_numpy(inp).cuda(),requires_grad=False)
# x=nt.forward(inp)






