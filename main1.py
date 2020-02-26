import keras
import numpy as np

new_model = keras.models.load_model('cb.h5')

from keras.preprocessing import image
for i in range(35):
  pred = image.load_img('./healthy/'+str(i)+'.jpg', target_size=(224, 224))
  pred = image.img_to_array(pred)
  pred = np.expand_dims(pred, axis=0)

  result = new_model.predict(pred)
  if np.argmax(result[0]==0):
      print("Healthy")
  else:
      print("Rice blast!")
