
# coding: utf-8

# In[1]:


import png
import os
import numpy as np
import matplotlib.pyplot as plt
import keras


# # Importing data 

# In[2]:


def import_train_images():
    train_images=[]
    files=os.listdir('/home/pedro/Documents/AutoEncoder/train')
    print(files)
    for file in files:
        string='/home/pedro/Documents/AutoEncoder/train/'+file
        image=png.Reader(filename=string)
        w,h,pixels,metadata=image.read_flat()
        image=np.array(pixels)
        if len(image)==226800:
            image=image[0:139320]                  #cut end of big images
        image=image.reshape((258,540))
        train_images.append(image)
    return(train_images)


# In[3]:


def import_train_labels():
    train_labels=[]
    files=os.listdir('/home/pedro/Documents/AutoEncoder/train_cleaned')
    print(files)
    for file in files:
        string='/home/pedro/Documents/AutoEncoder/train_cleaned/'+file
        image=png.Reader(filename=string)
        w,h,pixels,metadata=image.read_flat()
        image=np.array(pixels)
        if len(image)==226800:
            image=image[0:139320]                  #cut end of big images
        image=image.reshape((258,540))
        train_labels.append(image)
    return(train_labels)   


# In[4]:


train_images=import_train_images()


# In[5]:


train_labels=import_train_labels()


# In[6]:


train_images[1]


# In[7]:


plt.imshow(train_images[50], cmap='gray')


# In[8]:


plt.imshow(train_labels[50], cmap='gray')


# # Normalizing data

# In[9]:


train_images=np.array(train_images)
train_images=train_images.astype('float32')/255
train_labels=np.array(train_labels)
train_labels=train_labels.astype('float32')/255


# In[10]:


train_images[1]


# # Convolutional Input

# In[11]:


from keras import backend as K


# In[12]:


rows = 258
cols = 540
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, rows, cols)
    train_labels = train_labels.reshape(train_labels.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    train_images = train_images.reshape(train_images.shape[0], rows, cols, 1)
    train_labels = train_labels.reshape(train_labels.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)


# In[13]:


input_shape


# # Creating Model

# In[14]:


from keras import models
from keras import layers


# In[15]:


network = models.Sequential()
network.add(layers.Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
network.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Dropout(0.25))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(1, (3, 3), activation='relu',padding='same'))


# In[16]:


network.summary()


# In[17]:


network2 = keras.utils.multi_gpu_model(network,gpus=2) #model for 2 GPUs


# # Compiling Model

# In[18]:


sgd=keras.optimizers.SGD(momentum=0.8)
network.compile(optimizer=sgd,loss=keras.losses.mean_squared_error,metrics=['accuracy'])
network2.compile(optimizer=sgd,loss=keras.losses.mean_squared_error,metrics=['accuracy'])


# In[ ]:


earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=20)
history = network2.fit(train_images, train_labels,batch_size=2,epochs=100,validation_split=0.2, callbacks=[earlystop])

