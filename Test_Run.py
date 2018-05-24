
# coding: utf-8

# # Importing Test Data

# In[9]:


import keras
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import png


# In[10]:


def import_test_images():
    test_images=[]
    files=os.listdir('/home/pedro/Documents/AutoEncoder/test')
    print(files)
    for file in files:
        string='/home/pedro/Documents/AutoEncoder/test/'+file
        image=png.Reader(filename=string)
        w,h,pixels,metadata=image.read_flat()
        image=np.array(pixels)
        if len(image)==226800:
            image=image.reshape((420,540))              
        else:
            image=image.reshape((258,540))
        test_images.append(image)
    return(test_images)


# In[11]:


def Data_split(img,lab):
    small_images=[]
    small_labels=[]
    big_images=[]
    big_labels=[]
    for i in range(len(img)):
        if len(img[i])==258:
            small_images.append(img[i])
            small_labels.append(lab[i])
        if len(img[i])==420:
            big_images.append(img[i])
            big_labels.append(lab[i])
    small_images=np.array(small_images)
    small_images=small_images.astype('float32')/255
    small_labels=np.array(small_labels)
    small_labels=small_labels.astype('float32')/255
    big_images=np.array(big_images)
    big_images=big_images.astype('float32')/255
    big_labels=np.array(big_labels)
    big_labels=big_labels.astype('float32')/255
    from keras import backend as K
    rows = 258
    cols = 540
    if K.image_data_format() == 'channels_first':
        small_images = small_images.reshape(small_images.shape[0], 1, rows, cols)
        small_labels = small_labels.reshape(small_labels.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        small_images = small_images.reshape(small_images.shape[0], rows, cols, 1)
        small_labels = small_labels.reshape(small_labels.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    rows = 420
    cols = 540
    if K.image_data_format() == 'channels_first':
        big_images = big_images.reshape(big_images.shape[0], 1, rows, cols)
        big_labels = big_labels.reshape(big_labels.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        big_images = big_images.reshape(big_images.shape[0], rows, cols, 1)
        big_labels = big_labels.reshape(big_labels.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return((small_images,small_labels,big_images,big_labels))


# In[12]:


test_images=import_test_images()


# In[13]:


test_images[1]


# In[14]:


plt.imshow(test_images[50], cmap='gray')


# # Normalizing Test Data

# In[15]:


(test_small_images,ignore,test_big_images,ignore2)=Data_split(test_images,test_images)
#2 values ignored because we have no test labels


# In[16]:


test_small_images[1]


# # Import Trained Network

# In[17]:


network=keras.models.load_model('trained_network_v2_multi_size.h5')


# In[18]:


network.summary()


# # Run Test

# In[20]:


test_big_result=network.predict(test_big_images)


# In[21]:


test_small_result=network.predict(test_small_images)


# In[22]:


test_big_result[1]


# In[23]:


example=test_small_result[1]
example=example*255
example=example.astype('uint8')
example=example.reshape(258,540)


# In[24]:


plt.imshow(example, cmap='gray')


# In[25]:


pre_example=test_small_images[1]
pre_example=pre_example*255
pre_example=pre_example.astype('uint8')
pre_example=pre_example.reshape(258,540)


# In[26]:


plt.imshow(pre_example, cmap='gray')


# # convert Test result to the csv format accepted by Kaggle

# Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:
# 
# id,value
# 1_1_1,1
# 1_2_1,1
# 1_3_1,1
# etc.
