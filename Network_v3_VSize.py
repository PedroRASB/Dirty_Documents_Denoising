
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
    print(len(files)) #144
    for file in files:
        string='/home/pedro/Documents/AutoEncoder/train/'+file
        image=png.Reader(filename=string)
        w,h,pixels,metadata=image.read_flat()
        image=np.array(pixels)
        if len(image)==226800:
            image=image.reshape((420,540))              
        else:
            image=image.reshape((258,540))
        train_images.append(image)
    validation_images=train_images[0:28] #20%
    train_images=train_images[28::]
    return((train_images,validation_images)) #20%


# In[3]:


def import_train_labels():
    train_labels=[]
    files=os.listdir('/home/pedro/Documents/AutoEncoder/train_cleaned')
    for file in files:
        string='/home/pedro/Documents/AutoEncoder/train_cleaned/'+file
        image=png.Reader(filename=string)
        w,h,pixels,metadata=image.read_flat()
        image=np.array(pixels)
        if len(image)==226800:
            image=image.reshape((420,540))              
        else:
            image=image.reshape((258,540))
        train_labels.append(image)
    validation_labels=train_labels[0:28] #20%
    train_labels=train_labels[28::]
    return((train_labels,validation_labels))   


# In[4]:


(train_images,validation_images)=import_train_images()


# In[5]:


(train_labels,validation_labels)=import_train_labels()


# In[6]:


train_images[1]


# In[7]:


len(train_images)


# In[8]:


plt.imshow(train_images[50], cmap='gray')


# In[9]:


plt.imshow(train_labels[50], cmap='gray')


# # Normalizing data

# In[10]:


len(train_images[0])


# In[11]:


def Data_split(img,lab): #split data depending on image size (540 rows or 258 rows)
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


(small_images,small_labels,big_images,big_labels)=Data_split(train_images,train_labels)


# In[13]:


(small_images_val,small_labels_val,big_images_val,big_labels_val)=Data_split(validation_images,validation_labels)


# In[14]:


small_images


# In[15]:


len(small_images)


# In[16]:


len(big_images)


# In[17]:


len(small_images_val)


# In[18]:


len(big_images_val)


# In[19]:


from random import shuffle


# In[20]:


def Batch_generator(s_img,b_img,s_lab,b_lab):
    while True:
        small_images_s=[]
        big_images_s=[]
        small_labels_s=[]
        big_labels_s=[]
        s=list(range(len(s_img)))
        b=list(range(len(b_img)))
        shuffle(s)
        shuffle(b)
        for i in range(len(s_img)):
            k=s[i]
            small_images_s.append(s_img[k])
            small_labels_s.append(s_lab[k])
        for i in range(len(b_img)):
            k=b[i]
            big_images_s.append(b_img[k])
            big_labels_s.append(b_lab[k])
        le=len(s_img)+len(b_img)
        if le>=72: #training data recieved
            a=8
            b=15
            c=2
            d=4
        if le<=72: #validation data recieved
            a=3
            b=4
            c=1
            d=2
        for i in range(a):  #8 batches created
            i=i*5 #batch size of 5
            if i==35:
                j=i+c
            else:
                j=i+5
            batch_image=np.array(small_images_s[i:j])
            batch_label=np.array(small_labels_s[i:j])
            yield (batch_image,batch_label)
        for i in range(b): #15 batches created
            i=i*5
            if i==95:
                j=i+d
            else:
                j=i+5
            batch_image=np.array(big_images_s[i:j])
            batch_label=np.array(big_labels_s[i:j])
            yield (batch_image,batch_label)
            


# # Creating Model

# In[21]:


from keras import backend as K
if K.image_data_format() == 'channels_first':
    input_shape = (1, None, 540)
else:
    input_shape = (None, 540, 1)


# In[22]:


input_shape


# In[23]:


from keras import models
from keras import layers


# In[24]:


network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=input_shape))
network.add(layers.MaxPooling2D((2, 2), padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
network.add(layers.UpSampling2D((2, 2)))
network.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))


# In[25]:


network.summary()


# In[26]:


network2 = keras.utils.multi_gpu_model(network,gpus=2) #model for 2 GPUs


# # Compiling Model

# In[27]:


network.compile(optimizer='adadelta', loss='binary_crossentropy')
network2.compile(optimizer='adadelta', loss='binary_crossentropy')


# # Training Model

# In[28]:


earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10)
gen=Batch_generator(small_images,big_images,small_labels,big_labels) #need validation generator
val=Batch_generator(small_images_val,big_images_val,small_labels_val,big_labels_val)
history = network2.fit_generator(generator=gen,steps_per_epoch=23,epochs=100,validation_data=val,validation_steps=7,callbacks=[earlystop])


# # Saving Model and training results

# In[ ]:


import h5py               #save the network for 1 gpu
network.set_weights(network2.get_weights())
network.save('trained_network_v3_multi_size.h5')


# In[ ]:


result=network2.predict(np.array([big_images_val[5]]))


# In[ ]:


result


# In[ ]:


result=result*255
result=result.astype('uint8')
result=result.reshape(420,540)


# In[ ]:


result


# In[ ]:


plt.imshow(result, cmap='gray')


# # Importing Test Data

# In[ ]:


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


# In[ ]:


test_images=import_test_images()


# In[ ]:


test_images[1]


# In[ ]:


plt.imshow(test_images[50], cmap='gray')


# # Normalizing Test Data

# In[ ]:


test_images=np.array(test_images)
test_images=test_images.astype('float32')/255


# In[ ]:


test_images[1]


# # Creating test input

# In[ ]:


rows = 258
cols = 540
if K.image_data_format() == 'channels_first':
    test_images = test_images.reshape(test_images.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    test_images = test_images.reshape(test_images.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)


# # Running network on test data

# In[ ]:


test_result=network2.predict(test_images)


# In[ ]:


test_result[1]


# In[ ]:


example=test_result[70]
example=example*255
example=example.astype('uint8')
example=example.reshape(258,540)


# In[ ]:


plt.imshow(example, cmap='gray')


# In[ ]:


pre_example=test_images[1]
pre_example=pre_example*255
pre_example=pre_example.astype('uint8')
pre_example=pre_example.reshape(258,540)


# In[ ]:


plt.imshow(pre_example, cmap='gray')


# In[ ]:


small_images_s=[]
big_images_s=[]
small_labels_s=[]
big_labels_s=[]
s=list(range(len(small_images)))
b=list(range(len(big_images)))
shuffle(s)
shuffle(b)
for i in range(len(small_images)):
    k=s[i]
    small_images_s.append(small_images[k])
    small_labels_s.append(small_labels[k])
for i in range(len(big_images)):
    k=b[i]
    big_images_s.append(big_images[k])
    big_labels_s.append(big_labels[k])


# In[ ]:


big_images_s


# In[ ]:


np.shape(big_images_s)


# In[ ]:


big_images_s[0:5]


# In[ ]:


np.shape(small_images_s[10:11])


# In[ ]:


big_images[0]


# In[ ]:


big_images[0:5]


# In[ ]:


for i in range(90):
    print(len(big_images_s[i]))


# In[ ]:


for i in range(90):
    print(len(big_labels_s[i]))


# In[ ]:


batch_image=np.array(small_images_s[0:5])


# In[ ]:


batch_image


# In[ ]:


37/5

