#!/usr/bin/env python
# coding: utf-8

# In[3]
import numpy as np
import keras
size = (224, 224, 3)
size1 = (224, 224)

vgg19 = VGG19(input_shape=size, include_top=False, weights='imagenet')

for layer in vgg19.layers:
    layer.trainable = False

import cv2

# # def func(image):
# # 	# hls = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 	k = np.zeros((224, 224, 3))
# # 	k[:, :, 1] = image[:,:,1]
# # 	print(img.shape)
# # 	return img




# # In[4]:sssss


for i, layer in enumerate(vgg16.layers):
    print(str(i)+" "+layer.__class__.__name__, layer.trainable)


# # # In[5]:


from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

num_classes =2

l = vgg16.output
l = Flatten(name='flatten')(l)
fc1 = Dense(256, activation='relu')(l)
fc1 = Dropout(0.3)(fc1)
fc_head = Dense(num_classes, activation='softmax')(fc1)

model = Model(inputs = vgg16.input, outputs=fc_head)

model.summary()

model = keras.models.load_model('save_vgg19.h5')


# In[10]:


n_train = 1404
n_validation = 145
epochs = 30
batch_size = 32


from keras.preprocessing.image import ImageDataGenerator

train_dir = './RiceDiseaseDataset/train'
validation_dir = './RiceDiseaseDataset/validation'

train_datagen = ImageDataGenerator(rescale=1./255
                                  rotation_range=30,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.15,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  # preprocessing_function=func,
                                  fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=size1,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                   target_size=size1,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')



# In[11]:


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# early_stop = EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=,
#                           verbose=1,
#                           restore_best_weights=True)

checkpoint = ModelCheckpoint('weights.{epoch:02d}.hdf5',
                            save_best_only=False,
                            verbose=1)

callbacks = [checkpoint]

model.compile(loss='categorical_crossentropy',
               optimizer=Adam(lr=1e-3), metrics=['accuracy'])

hostory = model.fit_generator(train_generator,
                             steps_per_epoch=n_train//batch_size,
                              epochs=epochs,
                             callbacks = callbacks,
                             validation_data = validation_generator,
                             validation_steps = n_validation//batch_size)

model.save('save_vgg19_next.h5')





