import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

image = imread('000342.png')
print(image.shape)
image_arr = image
print(image_arr.shape)
feature = np.reshape(image,((384 * 1248*4) ,1))
for i in feature:
    i -= 100
new_img = np.reshape(feature,(384,1248,4))
print(new_img.shape)
plt.imshow(new_img)
plt.show()
#print(image.shape, image)

