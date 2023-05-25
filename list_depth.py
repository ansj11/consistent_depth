import os, sys
import numpy as np
import argparse, time
from collections import Counter
import matplotlib, cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
import new_module
#from transforms import *

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
#import util
import random
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img


def Bgr2Yuv(image):
    yuv = np.clip(image.copy(),0,255).astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    image = np.clip(yuv, 0, 255)
    return image.astype('uint8')

def depth_prediction(image, net):
    w = image.shape[0]
    h = image.shape[1]
    image = cv2.resize(image, (128,128),interpolation=cv2.INTER_LINEAR)
    #image = cv2.GaussianBlur(image,(5,5),0)
    image = Bgr2Yuv(image)
    image = image.astype(np.float32)
    image /= 255.0

    image = torch.from_numpy(image.astype('float32')).permute(2, 0, 1)
    image = image.cuda().unsqueeze(0)
    with torch.no_grad():
        output = net(image)[0]
        #output = torch.nn.functional.upsample(output, size=[w,h], mode='bilinear', align_corners=True)
    output = output.view(output.size(2),output.size(3)).data.cpu().numpy()
    return output

r = str(random.uniform(0,1))
def main():
    with torch.cuda.device(DEVICE_IDS[0]):   
        # network initialization
        print('Initializing model...')
        net = torch.load('./models/model-best.model',map_location = {'cuda:2':'cuda:1'})
        net = net.cuda()
        print('Done!')
        net.eval()
        i = 0
        r = str(round(random.uniform(0,1),3))
        with open('./img_list.txt') as f:
            for line in f.readlines():
                l = line[:-1]
                print (l)
                frame = cv2.imread(l)
                #cv2.imwrite('./images2/'+str(i)+'.jpg', frame)
                try:image_show = frame.copy()
                except: continue
                if random.uniform(0,1)<0.6: continue
                cv2.imwrite('./images2/'+str(i)+r+'.jpg', frame)
                image_show = cv2.resize(image_show,(128,128))
                output = depth_prediction(frame, net)
                ##output = output*1000
                ##output = output.astype('uint16')
                ##cv2.imwrite('./images2/'+str(i)+r+'.png', output)
                
                out = cv2.resize(output,(frame.shape[1],frame.shape[0]))
                matplotlib.image.imsave('./show/'+str(i)+'.png', output,vmax=min(13,np.max(output)))
                img = cv2.imread('./show/'+str(i)+'.png')
                img = np.concatenate((image_show,img),axis=1)
                cv2.imwrite('./show/'+str(i)+'.png', img)
                
                i+=1

if __name__ == '__main__':
    main()
