
# coding: utf-8

# In[ ]:


import os
import numpy as np
from scipy import misc
from shutil import copyfile
train_list = os.listdir('stage1_train')
train_list = [x for x in train_list if not x.startswith('.')]


# In[ ]:


#generate cumulative masks
for ii,f in enumerate(train_list):
    small_masks = os.listdir('stage1_train/'+f+'/masks')
    og_image = os.listdir('stage1_train/'+f+'/images')
    big_mask = 'placeholder'
    for s in small_masks:
        a=misc.imread('stage1_train/'+f+'/masks/'+s)
        if big_mask == 'placeholder':
            big_mask = a
        else:
            big_mask = big_mask + a            
    misc.imsave('train/y/'+'mask_'+str(ii)+'.png',big_mask)
    copyfile('stage1_train/'+f+'/images/'+og_image[0],'train/x/'+'img_'+str(ii)+'.png')


# In[ ]:


#extract 100 random patches from each image
from sklearn.feature_extraction import image
np.random.seed(0)

og_x_list = os.listdir('train/x/')
og_x_list = [ x for x in og_x_list if not x.endswith('.DS_Store') ]

for x in og_x_list:
    img_x = misc.imread('train/x/'+x)
    img_y = misc.imread('train/y/'+'mask_'+x[4:])
    
    img = np.dstack((img_x,img_y))
    
    patches = image.extract_patches_2d(img, (100+np.random.randint(low=0,high=100), 100+np.random.randint(low=0,high=100)),100,0) #50% of possible patches, random seed = 0
    dims = np.shape(patches)
    
    for i in range(dims[0]):
        mask = patches[i,:,:,-1]
        misc.imsave('train/y/'+'mask_'+x[4:-4]+'_'+str(i)+'.png',mask)
        
        patch = patches[i,:,:,0:dims[-1]-1]
        misc.imsave('train/x/'+'img_'+x[4:-4]+'_'+str(i)+'.png',patch)


# In[ ]:


#rotate and flip

xlist = os.listdir('./train/x/')
ylist = os.listdir('./train/y/')

for f in xlist:
    im = misc.imread('./train/x/'+f)
    im1 = np.rot90(im)
    im2 = np.rot90(im1)
    im3 = np.rot90(im2)
    f=f[:-4]
    misc.imsave('./train/x/'+f+'_r1.png',im1)
    misc.imsave('./train/x/'+f+'_r2.png',im2)
    misc.imsave('./train/x/'+f+'_r3.png',im3)
    
for f in ylist:
    im = misc.imread('./train/y/'+f)
    im1 = np.rot90(im)
    im2 = np.rot90(im1)
    im3 = np.rot90(im2)
    f=f[:-4]
    misc.imsave('./train/y/'+f+'_r1.png',im1)
    misc.imsave('./train/y/'+f+'_r2.png',im2)
    misc.imsave('./train/y/'+f+'_r3.png',im3)
    
xlist = os.listdir('./train/x/')
ylist = os.listdir('./train/y/')

for f in xlist:
    im = misc.imread('./train/x/'+f)
    im1 = np.flip_ud(im)
    f=f[:-4]
    misc.imsave('./train/x/'+f+'_flip.png',im1)
    
for f in ylist:
    im = misc.imread('./train/y/'+f)
    im1 = np.flip_ud(im)
    f=f[:-4]
    misc.imsave('./train/y/'+f+'_flip.png',im1)

