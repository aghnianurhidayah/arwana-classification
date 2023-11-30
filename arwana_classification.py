import numpy as np
import pickle
import tensorflow.keras.utils as image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

loaded_model = pickle.load(open('trained_model_arwana_classification.sav', 'rb'))

path = 'dataset_arwana/arwana_golden/G16.jpg'
img = image.load_img(path, target_size=(150,150))
plt.imshow(img)
plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = loaded_model.predict(images, batch_size=10)

if classes[0,0]:
  print('Arwana Golden')
elif classes[0,1]:
  print('Arwana Hitam')
elif classes[0,2]:
  print('Arwana Merah')
elif classes[0,3]:
  print('Arwana Silver')