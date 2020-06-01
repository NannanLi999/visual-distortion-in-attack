from scipy import signal
from scipy import ndimage
import cv2
import numpy as np
import os

def vis(img,name):
    subdir="/".join(name.split('/')[:-1])
    if not os.path.exists(subdir):
      os.mkdir(subdir)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(name.split('.')[0]+'.png',img)       

def fspecial_gauss(size, sigma):
    #Function to mimic the 'fspecial' gaussian MATLAB function
    
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
def compare_ssim(img1, img2, cs_map=False):
    #Return the Structural Similarity Map corresponding to input images img1 
    #and img2 (images are assumed to be uint8)
    
    #This function attempts to mimic precisely the functionality of ssim.m a 
    #MATLAB provided by the author's of SSIM
    #https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    
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
        return np.mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)))
  
     
