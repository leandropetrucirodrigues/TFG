import numpy as np
from numpy import mean
from PIL import Image, ImageDraw


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
    # img_2 = np.array(img_2)
    return img_2

imagens = read_npz('face_images.npz')

rostos = range(1,107)


for item in rostos:
    imagem = load_jpg(imagens,item)
    imagem.save("C:/Users/Leandro/Desktop/face-reco/Imagens/" + str(item) + ".png")
    