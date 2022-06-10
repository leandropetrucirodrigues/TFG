import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt


image_data = np.load('face_images.npz')
images = image_data['face_images']
images = np.transpose(images,(2,0,1))
imagem = []
for i,ima in enumerate(images):
    imagem.append(ima)

imagem = np.array(imagem)
imagem = np.expand_dims(imagem,axis =-1)
imagem = np.asarray(imagem)

# for i, image in enumerate(images):
#     if i not in na_idx:
#         cut_images.append(image)
# cut_images = np.array(cut_images)
# print(cut_images.dtype)
# cut_images = np.expand_dims(cut_images, axis = -1)
# print(cut_images.dtype)
# cut_images = np.asarray(cut_images)
# print(cut_images.dtype)

plt.figure(figsize = (20,20))

plt.imshow(imagem[5],cmap='gray')
plt.show()
    



# idx = random.sample(face_data.index.tolist(), 8)
# plt.figure(figsize = (20,20))
# for count, i in enumerate(idx):
#     image = cut_images[i]
#     plt.subplot(1, 8, count + 1)
#     plt.imshow(image, cmap = 'gray')
#     plt.scatter(face_data.iloc[i].tolist()[::2], face_data.iloc[i].tolist()[1::2])
    
    
# plt.show()