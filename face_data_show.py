import numpy as np
import cv2
from matplotlib import pyplot as plt

data= np.load('data.npy')  
print(data.shape)

# all names
print(data[:,data.shape[1]-1]) 

roi = data[10,0:data.shape[1]-1].reshape((150,150,3))
roi=np.array(roi, dtype=np.uint8)
plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')

plt.show()

