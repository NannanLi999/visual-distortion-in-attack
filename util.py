from scipy import signal
from scipy import ndimage
import cv2
import numpy as np
import os
import sys
import torch
from skimage.measure import compare_ssim
import skimage.color as color

IMPORT_LPIPS_SUCCESS=False
try:
   sys.path.append('./PerceptualSimilarity')
   import models as LPIPSmodel
   LPIPS= LPIPSmodel.PerceptualLoss(model='net-lin',net='alex',use_gpu=True,version='0.1')  
   IMPORT_LPIPS_SUCCESS=True
except Exception as e:
   print('Import libarary perceptual-similarity failed!')

def vis(img,name):
    subdir="/".join(name.split('/')[:-1])
    if not os.path.exists(subdir):
      os.mkdir(subdir)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(name.split('.')[0]+'.png',img)       

## im2tensor from PerceptualSimilarity/util/util.py
def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def fspecial_gauss(size, sigma):
    #Function to mimic the 'fspecial' gaussian MATLAB function
    
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
def compare_one_minus_ssim(img1, img2):
    score=1.0-compare_ssim(img1,img2, data_range=255, multichannel=True)
    return score
    
def compare_ciede2000(img1,img2):
        img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
        img1=np.float32(img1)/255.0
        img1= cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
        l1, a1, b1 = cv2.split(img1)
        l1,a1,b1=l1.flatten(),a1.flatten(),b1.flatten() 
              
        img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR) 
        img2=np.float32(img2)/255.0
        img2= cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
        l2, a2, b2 = cv2.split(img2) 
        l2,a2,b2=l2.flatten(),a2.flatten(),b2.flatten() 
        
        deta=0  
        for j in range(len(l1)):
          if (l1[j],a1[j],b1[j])==(l2[j],a2[j],b2[j]):
             t=0
          else:
             t=color.deltaE_ciede2000((l1[j],a1[j],b1[j]), (l2[j],a2[j],b2[j]))
          deta+=t
          
        return deta/len(l1)
"""             
def compare_one_minus_ssim_single(img1, img2, cs_map=False):
    #Return the Structural Similarity Map corresponding to input images img1 
    #and img2 (images are assumed to be uint8)
    
    #This function attempts to mimic precisely the functionality of ssim.m a 
    #MATLAB provided by the author's of SSIM
    #https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    #if len(img2.shape)==3 and img2.shape[2]==3:
    #   img2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
       
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return 1.0-np.mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)))
"""  
def compare_lpips(im1,im2):
      im1 = im2tensor(im1).cuda()
      im2 = im2tensor(im2).cuda()
      dist=torch.squeeze(LPIPS.forward(im1,im2))
      return dist.data.cpu().numpy()  
      
def GMSD(img1, img2):
    # ????,????
    # ?????quality_map????????????????????
    result_static, quality_map = cv2.quality.QualityGMSD_compute(img1, img2)
    # ????
    score = np.mean([i for i in result_static if (i != 0 and not np.isinf(i))])
    score = 0 if np.isnan(score) else score
    return score   
