
# coding: utf-8

# In[31]:


import png
import os
import numpy as np
import matplotlib.pyplot as plt
import keras


# # Importing data 

# In[32]:


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


# In[33]:


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


# In[34]:


train_images=import_train_images()


# In[35]:


train_labels=import_train_labels()


# In[36]:


train_images[1]


# In[37]:


train_images


# In[22]:


plt.imshow(train_images[50], cmap='gray')


# In[23]:


plt.imshow(train_labels[50], cmap='gray')


# # Normalizing data

# In[24]:


train_images=np.array(train_images)
train_images=train_images.astype('float32')/255
train_labels=np.array(train_labels)
train_labels=train_labels.astype('float32')/255


# In[25]:


train_images[1]


# In[26]:


train_images


# # Convolutional Input

# In[27]:


from keras import backend as K


# In[28]:


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


# In[29]:


input_shape


# In[30]:


train_images


# # Creating Model

# In[14]:


from keras import models
from keras import layers


# In[15]:


network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=input_shape))
network.add(layers.MaxPooling2D((2, 2), padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.MaxPooling2D((2, 2), padding='valid'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.UpSampling2D((2, 2)))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.UpSampling2D((2, 2)))
network.add(layers.ZeroPadding2D(padding=(1, 0), data_format=None))
network.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))


# In[16]:


network.summary()


# In[17]:


network2 = keras.utils.multi_gpu_model(network,gpus=2) #model for 2 GPUs


# # Compiling Model

# In[18]:


network.compile(optimizer='adadelta', loss='binary_crossentropy')
network2.compile(optimizer='adadelta', loss='binary_crossentropy')


# # Training Model

# In[19]:


earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10)
history = network2.fit(train_images, train_labels,batch_size=5,shuffle=True,epochs=1000,validation_split=0.2, callbacks=[earlystop])


# # Saving Model and training results

# In[20]:


import h5py               #save the network for 1 gpu
network.set_weights(network2.get_weights())
network.save('trained_network_v2.h5')


# In[27]:


result=network2.predict(np.array([train_images[50]]))


# In[28]:


result


# In[29]:


result=result*255
result=result.astype('uint8')
result=result.reshape(258,540)


# In[30]:


result


# In[31]:


plt.imshow(result, cmap='gray')


# # Importing Test Data

# In[58]:


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
            image=image[0:139320]                  #cut end of big images
        image=image.reshape((258,540))
        test_images.append(image)
    return(test_images)


# In[59]:


test_images=import_test_images()


# In[39]:


test_images[1]


# In[61]:


plt.imshow(test_images[50], cmap='gray')


# # Normalizing Test Data

# In[41]:


test_images=np.array(test_images)
test_images=test_images.astype('float32')/255


# In[42]:


test_images[1]


# # Creating test input

# In[43]:


rows = 258
cols = 540
if K.image_data_format() == 'channels_first':
    test_images = test_images.reshape(test_images.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    test_images = test_images.reshape(test_images.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)


# # Running network on test data

# In[46]:


test_result=network2.predict(test_images)


# In[49]:


test_result[1]


# In[51]:


example=test_result[1]
example=example*255
example=example.astype('uint8')
example=example.reshape(258,540)


# In[52]:


plt.imshow(example, cmap='gray')


# In[56]:


pre_example=test_images[1]
pre_example=pre_example*255
pre_example=pre_example.astype('uint8')
pre_example=pre_example.reshape(258,540)


# In[57]:


plt.imshow(pre_example, cmap='gray')

