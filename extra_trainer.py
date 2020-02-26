import numpy as np
import keras
size = (224, 224, 3)
size1 = (224, 224)
import cv2

num_classes =2

model = keras.models.load_model('save_vgg19.h5')

n_train = 1404
n_validation = 145
epochs = 30
batch_size = 32


from keras.preprocessing.image import ImageDataGenerator

train_dir = './RiceDiseaseDataset/train'
validation_dir = './RiceDiseaseDataset/validation'

train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=size1,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                   target_size=size1,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint('weights.{epoch:02d}.hdf5',
                            save_best_only=False,
                            verbose=1)


# early_stop = EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=3,
#                           verbose=1,
#                           restore_best_weights=True)

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
