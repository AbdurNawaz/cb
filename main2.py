import numpy as np
import keras
size = (224, 224, 3)
size1 = (224, 224)



from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization



chanDim = -1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=size))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model = Model(inputs = vgg16.input, outputs=fc_head)

model.summary()


n_train = 1404
n_validation = 145
epochs = 30
batch_size = 32


from keras.preprocessing.image import ImageDataGenerator        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        convert output probabilities to predicted class

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


early_stop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

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