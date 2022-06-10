import random
import pandas as pd
import numpy as np
from numpy import mean
from PIL import Image, ImageDraw
import face_recognition
import time

def read_npz(path_file_npz):
    image_data = np.load(path_file_npz)
    images = image_data['face_images']
    images = np.transpose(images,(2,0,1))
    imagem = []
    for i,ima in enumerate(images):
        imagem.append(ima)
    imagem = np.array(imagem)
    imagem = np.asarray(imagem)
    return imagem


def load_jpg(file_numpy_array,indice):
    img = Image.fromarray(file_numpy_array[indice])
    img_2 = img.convert("RGB")
    img_2 = np.array(img_2)
    return img_2


# Lê arquivo .npz com as imagens
imagens = read_npz('face_images.npz')


rostos = [7,37,93]

for item in rostos:
    imagem = load_jpg(imagens,item)
    pil_image = Image.fromarray(imagem)
    b = ImageDraw.Draw(pil_image)

    if item == 7:
        b.point((63.8999,36.9886),fill="blue")
        b.point((29.6188,38.2898),fill="blue")
        pil_image.save("C:/Users/Leandro/Desktop/face-reco/Photos/AWS_" +str(item) + ".png")

    elif item == 37:
        b.point((69.1211,37.1676),fill="blue")
        b.point((28.5455,36.9213),fill="blue")
        pil_image.save("C:/Users/Leandro/Desktop/face-reco/Photos/AWS_" +str(item) + ".png")

    elif item == 93:
        b.point((69.8778,35.8666),fill="blue")
        b.point((27.9002,34.5633),fill="blue")
        pil_image.save("C:/Users/Leandro/Desktop/face-reco/Photos/AWS_" +str(item) + ".png")




    