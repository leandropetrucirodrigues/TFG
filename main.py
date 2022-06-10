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



# Lê arquivo .CSV com os dados dos rostos
face_data =pd.read_csv('facial_keypoints.csv')

# Lê arquivo .npz com as imagens
imagens = read_npz('face_images.npz')



rostos = range(1,107)
resultados = []
rostos_nao_lidos = [2,6,7,10,13,15,24,25,26,34,36,38,39,43,48,50,52,53,56,62,63,65,66,68,70,71,72,73,75,76,80,81,83,85,87,88,89,90,94,96,97,101,102,103,104,105,106,]
n_imagens = 0

inicio = time.time()

for item in rostos:
    if item not in rostos_nao_lidos:
        # carregando imagem para analise
        imagem = load_jpg(imagens,item)
        # faces com pontos marcados
        face_landmarks_list = face_recognition.face_landmarks(imagem)
        n_imagens+=1
        if len(face_landmarks_list) == 0:
            rostos_nao_lidos.append(item)


        # listando as faces
        for face_landmarks in face_landmarks_list:


            pred_left_eye = list(mean(face_landmarks['right_eye'], axis=0)) 
            pred_right_eye = list(mean(face_landmarks['left_eye'], axis=0))

            face_marca = {
                'keypoint': item,
                'pred_left_eye_x' : pred_left_eye[0],
                'pred_left_eye_y' : pred_left_eye[1],
                'pred_right_eye_x' : pred_right_eye[0],
                'pred_right_eye_y' : pred_right_eye[1],
                'real_left_eye_x' : face_data['left_eye_center_x'][item],
                'real_left_eye_y' : face_data['left_eye_center_y'][item],
                'real_right_eye_x' : face_data['right_eye_center_x'][item],
                'real_right_eye_y' : face_data['right_eye_center_y'][item],
            }
            resultados.append(face_marca)
        #Desenha na imagem e abrir um arquivo de imagem

        # if item == 7 or item == 37 or item == 93:
        #     pil_image = Image.fromarray(imagem)
        #     pil_image2 = Image.fromarray(imagem)
        #     d = ImageDraw.Draw(pil_image)
        #     a = ImageDraw.Draw(pil_image2)


        #     d.point((pred_left_eye[0],pred_left_eye[1]),fill="red")
        #     d.point((pred_right_eye[0],pred_right_eye[1]),fill="red")

        #     a.point((face_data['left_eye_center_x'][item],face_data['left_eye_center_y'][item]),fill="yellow")
        #     a.point((face_data['right_eye_center_x'][item],face_data['right_eye_center_y'][item]),fill="yellow")

        #     pil_image.save("C:/Users/Leandro/Desktop/face-reco/Photos/dlib_" +str(item) + ".png")
        #     pil_image2.save("C:/Users/Leandro/Desktop/face-reco/Photos/Keypoints_" +str(item) + ".png")
            


fim = time.time()
print(n_imagens)
print((fim - inicio)/n_imagens)

olhos = pd.DataFrame(data = resultados)
olhos.to_csv('olhos.csv')
olhos

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, median_absolute_error

y_true = olhos[['real_left_eye_x',	'real_left_eye_y',	'real_right_eye_x',	'real_right_eye_y']].values
y_pred = olhos[['pred_left_eye_x',	'pred_left_eye_y',	'pred_right_eye_x', 'pred_right_eye_y']].values

print('MAE =',mean_absolute_error(y_true, y_pred))
print('MSE =',mean_squared_error(y_true, y_pred))
print('MAPE =',mean_absolute_percentage_error(y_true, y_pred))
print('R² =',r2_score(y_true, y_pred))
print('MedAE =',median_absolute_error(y_true, y_pred))


