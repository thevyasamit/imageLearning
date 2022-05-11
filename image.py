import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import laplace, gaussian
from skimage.io import imread, imshow

image = imread('000342.png')
print(image.shape)
image_arr = image
print(image_arr.shape)
feature = np.reshape(image,((384 * 1248 * 4) ,1))
for i in feature:
    i += 200
new_img = np.reshape(feature,(384,1248,4))
print(new_img.shape)

imgLaplace = laplace(image)
imgGaussian = gaussian(image)



plt.imshow(imgGaussian)
plt.show()
#print(image.shape, image)
'''
The image is basically an array on numbers.
To identify and detect features from image we often have to manipulate the image. 
In various deep learning models the fearures such as edges, contours, dimensions, etc plays an important role as they are input to the network.
'''

''' 
Edges in the image: The edge is a point or pixel where there is a big change in its value from its neighbors.
Since image is an array of pixels to manipulate with it we need small arrays as well and there are kernels (small arrays)
available using which we can find the edges in the image. The skimage has a lot of filters and I have tied using gaussian and laplace and
I thin laplace worked better for this image. 
'''

'''
The image used here is generated using CARLA simulator.
'''

