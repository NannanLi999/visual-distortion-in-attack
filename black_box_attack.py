import tensorflow as tf
import numpy as np
import math
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import random
import sys
import cv2

from util import compare_one_minus_ssim, compare_lpips   

class Black_Box_Attack:
    def __init__(self, sess, network,max_iterations = 10000,outer_iterations=5,lambda_iterations=1,batch_size=1, epsilon=0.05, 
                lambda_=10.0, metric='1-SSIM',learning_rate = 0.01,q=0.01,N=1,minval=0.0,maxval=1.0):
                 
        image_size, num_channels, num_labels = network.image_size, network.num_channels, network.num_labels
        self.sess = sess
        self.network=network
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations  
        self.outer_iterations=outer_iterations  
        self.lambda_iterations=lambda_iterations 
        self.batch_size =batch_size
        self.num_channels=num_channels
        self.q=q
        self.lambda_=lambda_
        self.metric=metric
        self.epsilon=tf.placeholder(tf.float32,[batch_size])
        self.minval=minval
        self.maxval=maxval
        self.tile_size=2
        self.network=network
        self.num_choice=2*N+1
        
        self.image_size=image_size
        self.tiled_image_size=image_size//self.tile_size
        
        self.timg = tf.placeholder(tf.float32,(batch_size,image_size,image_size,num_channels))
        self.baseline=tf.placeholder(tf.float32,[batch_size])
        self.best_choice=tf.placeholder(tf.int32,(batch_size,self.tiled_image_size,self.tiled_image_size))
        
        ## object-mask
        self.shape_mask=tf.placeholder(tf.float32, [batch_size,image_size,image_size])
        ## downsample object-mask
        shape_mask=tf.nn.conv2d(tf.expand_dims(self.shape_mask,3),tf.ones((self.tile_size,self.tile_size,1,1)),[1,self.tile_size,self.tile_size,1],padding='SAME')/(self.tile_size**2)
        shape_mask=shape_mask[:,:,:,0]
        ## out-of-object mask
        shape_mask=1.0-tf.to_float(tf.greater(shape_mask,0.5))
        
        self.cur_L=tf.placeholder(tf.float32, [batch_size])
        self.choice_plhd=tf.placeholder(tf.int32, [batch_size,self.tiled_image_size,self.tiled_image_size])

        
        uniform=np.zeros((batch_size*self.tiled_image_size*self.tiled_image_size,self.num_choice))            
        self.distri_logits=tf.Variable(uniform,dtype=tf.float32)  
        self.distri=tf.nn.softmax(self.distri_logits,1)
        
        ## initilize noise value to be 0
        self.choice_init=N*tf.ones((batch_size,self.tiled_image_size,self.tiled_image_size),dtype=tf.int32)
        noise_init=tf.to_float(self.choice_init)/((0.5/self.epsilon)*(self.num_choice-1))-self.epsilon          
        _,self.img_init=self.get_adv_img(noise_init,shape_mask=shape_mask)
               
        ## a square-shaped update of noise. Resample proportion q of the noise using rand_mask
        window_size=max(1,int(self.tiled_image_size*math.sqrt(self.q)))
        window=tf.ones([batch_size,window_size,window_size],dtype=tf.int32)
        center_row=tf.random_uniform([1],minval=0,maxval=tf.maximum(1,self.tiled_image_size-window_size+1),dtype=tf.int32)
        center_col=tf.random_uniform([1],minval=0,maxval=tf.maximum(1,self.tiled_image_size-window_size+1),dtype=tf.int32)
        def cond(cr,cc):
           rand_mask=tf.pad(window,paddings=[[0,0],[cr[0],self.tiled_image_size-cr[0]-window_size],[cc[0],self.tiled_image_size-cc[0]-window_size]],mode='CONSTANT')
           return tf.equal(tf.reduce_sum(tf.to_float(rand_mask)*shape_mask),0)
        def body(cr,cc):
            cr=tf.random_uniform([1],minval=0,maxval=tf.maximum(1,self.tiled_image_size-window_size+1),dtype=tf.int32)
            cc=tf.random_uniform([1],minval=0,maxval=tf.maximum(1,self.tiled_image_size-window_size+1),dtype=tf.int32)
            return cr,cc
        center_row,center_col=tf.while_loop(cond,body,[center_row,center_col])
        rand_mask=tf.pad(window,paddings=[[0,0],[center_row[0],self.tiled_image_size-center_row[0]-window_size],[center_col[0],self.tiled_image_size-center_col[0]-window_size]],mode='CONSTANT')
        rand_choice=tf.reshape(tf.to_int32(tf.squeeze(tf.multinomial(self.distri_logits,1),1)),[batch_size,self.tiled_image_size,self.tiled_image_size])
        self.cur_choice=(1-rand_mask)*self.best_choice+rand_mask*rand_choice
        
        noise=tf.to_float(self.cur_choice)/((0.5/self.epsilon)*(self.num_choice-1))-self.epsilon  
        self.noise,self.newimg=self.get_adv_img(noise,shape_mask=shape_mask)
             
        self.detaL=self.cur_L-self.baseline
        index=tf.one_hot(tf.reshape(self.choice_plhd,[-1]),self.num_choice,dtype=tf.float32)
        self.logprob=tf.reshape(tf.log(tf.clip_by_value(tf.reduce_sum(self.distri*index,1),1e-15,1.0)),[batch_size,self.tiled_image_size,self.tiled_image_size])
        self.mask=tf.to_float(tf.not_equal(self.best_choice,tf.to_int32(self.choice_plhd))) ## mask for resampled noise
        loss=self.logprob*self.mask*shape_mask*tf.expand_dims(tf.expand_dims(self.detaL,1),2)
        self.loss=tf.reduce_sum(loss)/(1e-10+tf.reduce_sum(self.mask*shape_mask))
        
        ## Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss, var_list=[self.distri_logits])#distri_logits
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.init = tf.variables_initializer(var_list=[self.distri_logits]+new_vars)#
    
    def preprocess(self,imgs):
        """
        input:  
           imgs:  images of range [0,1]
        output:  
           nimgs: preprocessed images for pytorch
        """
        ones=np.ones((imgs.shape[0],imgs.shape[1],imgs.shape[2],1),dtype=np.float32)
        nimgs=imgs-np.concatenate([0.485*ones, 0.456*ones, 0.406*ones],axis=3)
        nimgs/=np.concatenate([0.229*ones,0.224*ones, 0.225*ones],axis=3)
        nimgs=np.transpose(nimgs,[0,3,1,2])
        nimgs=torch.from_numpy(nimgs)
        return nimgs
            
    def get_adv_img(self,noise,shape_mask=None):  
        """
        Get perturbed images.
        
        input:   
            noise:      tiled noise map, (self.batch_size,self.tield_size,self.tield_size)
            shape_mask: out-of-object mask
        output:  
            noise:    noise map of size (image_size, image_size,image_channel)
            img:      perturbed images
        """      
        
        
        if shape_mask is not None:            
            noise=noise*shape_mask
        noise=tf.expand_dims(noise,3)
            
        ## upsample the noise map to image_size*image_size    
        noise=tf.nn.conv2d_transpose(noise,tf.ones((self.tile_size,self.tile_size,1,1)),[self.batch_size,self.image_size,self.image_size,1],[1,self.tile_size,self.tile_size,1],padding='SAME')
        noise=tf.tile(noise,[1,1,1,self.num_channels])            
        img=tf.clip_by_value(noise + self.timg,self.minval,self.maxval)
            
        return noise,img
    
    
    def get_loss_from_network(self,imgs,labels): 
       """
       Compute reward of images using pretrained networks on pytorch.
       
       input:   
            imgs:     perturbed images
            labels:   ground truth lable
       output:  
            loss:     max(0,{f(x+nosie)_y}-{max_{k!=y}}({f(x+noise)_k}))
            logits:   output logits  
             
        """    
       input_tensor=self.preprocess(imgs) 
       if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            labels_tensor=torch.from_numpy(labels).to('cuda')  
       with torch.no_grad():         
            output=torch.log(nn.functional.softmax(self.network.predict(input_tensor),dim=1))
       real=torch.sum(output*labels_tensor,dim=1).data.cpu().numpy()
       other=torch.max(output*(1-labels_tensor)-labels_tensor*10000,dim=1)[0].data.cpu().numpy()
       loss=np.maximum(0.0, real-other)
       logits=output.data.cpu().numpy()
       return loss,logits
       
    def attack(self, imgs, labels,shape_masks):
        """
        Run the attack on all images and labels.
        input:
           imgs: orginal images
           labels: ground truth labels
        output:
           shape_masks: object masks
        """
        r = {'attack':[],'flag':[],'num_query':{'ave':[]}}
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            attack,flags,num_query=self.attack_batch(imgs[i:i+self.batch_size], labels[i:i+self.batch_size],shape_masks[i:i+self.batch_size])
            r['attack'].extend(attack)
            r['flag'].extend(flags)
            r['num_query']['ave'].extend(num_query['ave'])
        return r
    
       
    def attack_batch(self, batch, batchlab ,batchmask):
        """
        Run the attack on a batch of images and labels.
        input:
           imgs: a batch of orginal images
           labels: a batch of ground truth labels
        output:
           shape_masks: a batch of object masks
        """
        
        def is_successful_attack(scores,label_vector):
            return np.argmax(scores)!=np.argmax(label_vector)
            
        if self.metric=='1-SSIM':
            compute_distance_metric=compare_one_minus_ssim
        elif self.metric=='LPIPS':
            compute_distance_metric=compare_lpips
        else:
            print('Undefinned perceptual distance metric!')
            sys.exit()
                 
        batch_size = self.batch_size
        image_size=batch[0].shape[1]
        
        o_flags=np.array([-1]*batch_size,dtype=np.int32)
        o_bestdistance=np.array([1e10]*batch_size,dtype=np.float32)
        bestattck=[np.zeros(batch[0].shape)]*batch_size
                                             
        all_num_query=[[] for _ in range(batch_size)]
        
        _,scores_ori=self.get_loss_from_network(batch,batchlab)
        wrong=np.not_equal(np.argmax(scores_ori,1),np.argmax(batchlab,1))
        o_flags=(1-wrong.astype(np.int32))*o_flags
        epsilons=0.05*np.ones(batch_size)
        for outer_i in range(self.outer_iterations):
          ## flags: 0->not correctly classified, 1-> succesful attack, -1->corresctly classified sample
          flags=np.array([-1]*batch_size,dtype=np.int32)
          flags=(1-wrong.astype(np.int32))*flags      
          
          lambda_=self.lambda_*np.ones(batch_size)
          lower_bound=np.zeros(batch_size)
          upper_bound=1000*np.ones(batch_size)
           
          for i_lambda in range(self.lambda_iterations):  
            ## initialize the model       
            self.sess.run(self.init)
            
            #reset flags
            flags=np.where(np.equal(flags,1),-1,flags)
            
            ## compute baseline b
            feed_dict={self.timg:batch,self.shape_mask:batchmask,self.epsilon:epsilons} 
            best_choice,nimg=self.sess.run([self.choice_init,self.img_init],feed_dict=feed_dict)
            baseline,scores=self.get_loss_from_network(nimg,batchlab )                
            
            distance=[]
            for e in range(batch_size):
               im1=(255*nimg[e]).astype(np.uint8)
               im2=(255*batch[e]).astype(np.uint8)
               distance.append(compute_distance_metric(im1,im2))
            baseline+=lambda_*np.array(distance)   
            print('lambda:',lambda_)   
            
            for iteration in range(self.max_iterations): 
                  
              for e in range(batch_size):
                  if is_successful_attack(scores[e], batchlab[e]) and flags[e]==-1:                
                      flags[e]=1
                      o_flags[e]=1
                      if distance[e]<o_bestdistance[e]:                      
                         o_bestdistance[e]=distance[e]
                         bestattck[e] = nimg[e]
                         all_num_query[e].append(iteration+1)
                         
              ## early stop if all samples are misclassified
              if np.sum(np.equal(flags,-1))==0:
                   break    
              
              ## compute L     
              feed_dict={self.timg:batch,self.shape_mask:batchmask,self.baseline:baseline,self.best_choice:best_choice,self.epsilon:epsilons}                           
              cur_choice,nimg=self.sess.run([self.cur_choice,self.newimg],feed_dict=feed_dict)                      
              cur_L,scores=self.get_loss_from_network(nimg,batchlab)                  
              distance=[]
              for e in range(batch_size):
                      im1=(255*nimg[e]).astype(np.uint8)
                      im2=(255*batch[e]).astype(np.uint8)
                      distance.append(compute_distance_metric(im1,im2))
              cur_L+=lambda_*np.array(distance)                    
              
              ## gradient descent      
              feed_dict.update({self.cur_L:cur_L,
                         self.choice_plhd:cur_choice
                         })
              _, l,  detaL= self.sess.run([self.train, self.loss,self.detaL],feed_dict=feed_dict)
  
              ## update baseline b
              for e in range(batch_size):
                 if detaL[e]<0:
                   best_choice[e]=cur_choice[e]
                   baseline[e]+=detaL[e]
                   
              # print out the losses every 10%
              if iteration%(self.max_iterations//10) == 0:
                      print(outer_i, iteration,'loss:',l,', baseline:',baseline[:5],distance[:5])                  
            for e in range(batch_size):   
               if flags[e]==1:
                  lower_bound[e] = max(lower_bound[e],lambda_[e])
                  lambda_[e] = (lower_bound[e] + upper_bound[e])/2
               else:
                 upper_bound[e] = min(upper_bound[e],lambda_[e])
                 lambda_[e] = (lower_bound[e] + upper_bound[e])/2        
        for e in range(batch_size):
            if o_flags[e]==-1:
               print('sample '+str(e)+' failed')
               
        num_query={'ave':[]}
        for x in all_num_query:
           if len(x)>0:
              num_query['ave'].append(x[-1])

        return bestattck,o_flags,num_query