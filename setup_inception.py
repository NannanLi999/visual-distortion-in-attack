import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from PIL import Image

import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import pickle

class INCEPTION:
    def __init__(self,image_root,bboxroot,num_images=1000,use_mask=False):
        
        self.num_class=1000
        self.num_images=num_images
        self.image_size=300
        
        self.synset2label=self.get_synset2label('./imagenet_lsvrc_2015_synsets.txt')
        self.label2human=self.get_label2human('./imagenet_metadata.txt')
        
        
        test_data = []
        test_labels = []
        test_masks = []
        
        imglist=self.get_imglist(image_root) 
        self.imglist=[]
         
        if use_mask:
             all_masks=self.load_pascal_annotation(bboxroot)   
             
        preprocess = transforms.Compose([
                          transforms.Resize(300),
                          transforms.CenterCrop(300),
                            ])            
        
        for path in imglist:
           synset=path.split('/')[0] 
           imgname=path.split('/')[-1].split('.')[0]                     
                   
           label=np.zeros(self.num_class,dtype=np.float32)
           label[self.synset2label[synset]]=1.0
            
           img=Image.open(image_root+'/'+path).convert("RGB")
           img=preprocess(img)
           img=np.asarray(img,dtype=np.float32)/255.0
        
           if use_mask:
               mask=all_masks[imgname]
               if np.sum(mask)<self.image_size**2*0.1:#>self.image_size**2*0.7:
                  continue
               test_masks.append(mask)
               self.imglist.append(path)
           else:
               self.imglist.append(path)
              
           test_data.append(img)
           test_labels.append(label)
           if len(test_data)>=self.num_images:
              break
      
        self.test_data=np.stack(test_data,axis=0)
        self.test_labels=np.stack(test_labels,axis=0)
        if use_mask:
           self.test_masks=np.stack(test_masks,axis=0)
        assert len(self.test_data)==len(self.imglist)   
                
                   
    def get_imglist(self,image_root):
       imglist=[]
       synset_dir=[x for x in os.listdir(image_root)]
       for synset in synset_dir:
            imgs=os.listdir(image_root+'/'+synset)
            for name in imgs:
                 imglist.append(synset+'/'+name)
       try:
          imglist=np.random.choice(imglist,self.num_images,replace=False)
       except Exception as e:
           print('Not enough images! Using all images in the data directory.')
       return imglist
    
    def get_synset2label(self,path):
       synsets=open(path,'r')
       synset2label={}
       for i,synset in enumerate(synsets):
           synset2label[synset.strip()]=i
       synsets.close()
       return synset2label

    
    def get_label2human(self,path):
        synset_human_file=open(path,'r')  
        label2human={}
        for synset_human in synset_human_file:
           synset=synset_human.split('\t')[0].strip()
           human_string=synset_human.split('\t')[1].strip()  
           if self.synset2label.get(synset) is None:
               continue
           else:
               label=self.synset2label[synset]
               label2human[label]=human_string
        synset_human_file.close()
        return label2human
                
   
    def load_pascal_annotation(self,bboxroot):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        bboxfile=os.listdir(bboxroot)
        all_masks={}
        for file_path in bboxfile:
           tree = ET.parse(bboxroot+file_path)
           bboxname=tree.find('filename').text
           objs = tree.findall('object')    

           height=int(tree.find('size').find('height').text)
           width=int(tree.find('size').find('width').text)
           mask= np.zeros((height,width,1),dtype=np.uint8)
       
           # Load object bounding boxes into a data frame.
           for ix, obj in enumerate(objs):
              bbox = obj.find('bndbox')
              # Make pixel indexes 0-based
              x1 = int(bbox.find('xmin').text)
              y1 = int(bbox.find('ymin').text)
              x2 = int(bbox.find('xmax').text) 
              y2 = int(bbox.find('ymax').text)
              mask[y1:y2,x1:x2]=255
            
           """central_fraction=0.875
           edge_fraction=(1-central_fraction)/2.0
           mask=mask[int(height*edge_fraction):int(height*(1-edge_fraction)),int(width*edge_fraction):int(width*(1-edge_fraction)),:]  
           mask=cv2.resize(mask, (self.image_size, self.image_size))"""
           mask=cv2.resize(mask, (300, 300))
  
           mask=np.asarray(np.greater(mask,127),dtype=np.int32)
           all_masks[bboxname]=mask  
        return all_masks
        

class INCEPTIONModel:
    def __init__(self):
        self.num_channels = 3
        self.image_size = 300
        self.num_labels = 1000

        vgg= models.inception_v3(pretrained=True)
        vgg.eval()    
        if torch.cuda.is_available():
           self.predict=vgg.to('cuda')
"""
data=INCEPTION(num_images=10)
img=data.test_data[3]
#img=Image.open('/home/lnn/ILSVRC2012/val/n01440764/ILSVRC2012_val_00000293.JPEG')
preprocess = transforms.Compose([                        
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
#img_tensor=preprocess(img) 
imgs=img[None,:,:,:]
ones=np.ones((imgs.shape[0],imgs.shape[1],imgs.shape[2],1))
imgs-=np.concatenate([0.485*ones, 0.456*ones, 0.406*ones],axis=3)
imgs/=np.concatenate([0.229*ones,0.224*ones, 0.225*ones],axis=3)
imgs=np.transpose(imgs,[0,3,1,2])
input_tensor=torch.from_numpy(imgs)  
#input_tensor=img_tensor.unsqueeze(0)   
model=INCEPTIONModel()
if torch.cuda.is_available():
      input_tensor = input_tensor.to('cuda')  
     
with torch.no_grad():         
      output=nn.functional.softmax(model.predict(input_tensor),dim=1)
      s=output.data.cpu().numpy()[0]#
      print(np.max(s),np.argmax(s),np.argmax(data.test_labels[3]))
"""
   