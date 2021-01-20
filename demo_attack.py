
import numpy as np
import time
import random
import cv2
import pickle
import tensorflow as tf
import sys
import os

from setup_vggbn import VGGBN, VGGBNModel
from setup_resnet import RESNET, RESNETModel
from setup_inception import INCEPTION,INCEPTIONModel

from black_box_attack import Black_Box_Attack
from util import vis,compare_one_minus_ssim, compare_lpips,IMPORT_LPIPS_SUCCESS


IMAGE_DIR='demo/'
BBOX_DIR='/home/mm/LNN/ILSVRC2012/Annotation/val/'
GPU_ID='0' 

NETWORK='ResNet50'        ## name of the black-box network, could be 'InceptionV3', 'ResNet50' or 'VGG16bn'
USE_BBOX=False            ## whether to perform the out-of-object attack
MAX_ITERATION=10000       ## maximum number of iterations
NUM_TEST_IMAGES=1         ## number of test images
BATCH_SIZE=1              ## number of images per batch, it's required that NUM_TEST_IMAGES%batch_size==0 
NOISE_EPSILON=0.05        ## maximum noise value
PDIS_METRIC='1-SSIM'      ## perceptual distance metric, could be '1-SSIM' or 'LPIPS'
LAMBDA=10.0               ## lambda that controls the trade-off between visual distortion and query efficiency. Larger lambda leads to less visual distortion.
NUM_RANDOM_STARTS=1       ## number of random starts for the attack. 
LEARNING_RATE=0.01        ## learning rate for $\theta$
RESAMPLED_PROPORTION=0.01 ## the resampled propotion of the noise at each iteration, must be less or equal to 1
SAMPLE_FRQUENCY=1         ## sample frequency of the noise. The maximum value is 12.
SEARCH_STEPS=1           ## set search_steps>1 for dynamic lambda search     

     
def generate_data(data, samples):
    if USE_BBOX:
        masks=data.test_masks[:samples]
    else:
        masks=np.zeros((samples,data.test_data.shape[1],data.test_data.shape[2]))
    return  data.test_data[:samples],data.test_labels[:samples],masks


if __name__ == "__main__":
    
    ## config gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=False
    config.gpu_options.visible_device_list=GPU_ID
    
    ## specify the balck-box network and its dalaloader
    if NETWORK=='InceptionV3':
        data,network=INCEPTION(IMAGE_DIR,BBOX_DIR,num_images=NUM_TEST_IMAGES,use_mask=USE_BBOX),INCEPTIONModel()
    elif NETWORK=='ResNet50':
        data,network=RESNET(IMAGE_DIR,BBOX_DIR,num_images=NUM_TEST_IMAGES,use_mask=USE_BBOX),RESNETModel()   
    elif NETWORK=='VGG16bn': 
        data,network=VGGBN(IMAGE_DIR,BBOX_DIR,num_images=NUM_TEST_IMAGES,use_mask=USE_BBOX),VGGBNModel()
    else:
        print('UNKNOWN network!')
        sys.exit()
    print('model loaded')
    
    ## the minimum and maximum pixel value of the loaded images
    minval,maxval=0,1.0
   
    with tf.Session(config=config) as sess:
        attack = Black_Box_Attack(sess, network,max_iterations=MAX_ITERATION, lambda_iterations=SEARCH_STEPS,
                                  batch_size=BATCH_SIZE, outer_iterations=NUM_RANDOM_STARTS,epsilon=NOISE_EPSILON,
                                  lambda_=LAMBDA, metric=PDIS_METRIC,learning_rate=LEARNING_RATE, q=RESAMPLED_PROPORTION, N=SAMPLE_FRQUENCY, 
                                  minval=minval,maxval=maxval)
        

        inputs, labels,masks = generate_data(data, samples=NUM_TEST_IMAGES)
        timestart = time.time()
        results= attack.attack(inputs, labels,masks)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        
        ave_num_query=np.mean(results['num_query']['ave'])
        ## display average number of queries
        print('average number of queries:',ave_num_query)
        
        ## results['flag']: 0-> not correctly classified, 1->success, -1->failed
        success_mask=np.equal(results['flag'],1).astype(np.float32)
        corr_classified_mask=np.not_equal(results['flag'],0).astype(np.float32)
        success_rate=np.sum(success_mask)/np.maximum(1e-6,np.sum(corr_classified_mask))
        
        ## display success rate
        print('success rate:',success_rate)

        one_minus_ssim=[]
        lpips=[]
        ciede=[]
        for i in range(len(inputs)):
            ## save the results of successful attacks
            if results['flag'][i]==1:
                  
                  assert np.sum(np.greater(np.abs(inputs[i]-results['attack'][i]),NOISE_EPSILON+1e-5))==0
                  
                  ## convert adversarial example to uint8 type
                  img1=255.0*inputs[i]
                  img1=img1.astype(np.uint8)                 
                  img2=255.0*results['attack'][i]
                  img2=img2.astype(np.uint8)
                  
                  vis(img1,'ori/'+'_'.join(data.imglist[i].split('/')))
                  vis(img2,'adv/'+'_'.join(data.imglist[i].split('/')))
                                            
                  one_minus_ssim.append(compare_one_minus_ssim(img1,img2))  
                  if IMPORT_LPIPS_SUCCESS:
                      lpips.append(compare_lpips(img1,img2))       
                  ciede.append(compare_ciede2000(img1,img2))
        ## display results of the distance metric.
        if len(one_minus_ssim)>0:
           print('1-SSIM:',np.mean(one_minus_ssim))
           print('CIEDE:',np.mean(ciede))
           if len(lpips)>0:
               print('LPIPS:',np.mean(lpips))
        else:
           if np.sum(np.equal(flag,0))==len(flag):
               print('No images were correctly classified!')
           else:
               print('Attack failed for all samples.')
