
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import face_recognition

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

def random_faces(lista_rostos,qts_rostos):

    faces = random.sample(range(0,len(lista_rostos)),qts_rostos)

    return faces

def load_jpg(file_numpy_array,indice):

    img = Image.fromarray(file_numpy_array[indice])
    img_2 = img.convert("RGB")
    img_2 = np.array(img_2)

    return img_2

def calc_erro(X,Y,Xcsv,Ycsv):
    erroX = abs(X-Xcsv)
    erroY = abs(Y-Ycsv)

    erro = 'X:' + str(erroX) + ' - Y:' + str(erroY)

    return erro

def main_program(data_csv,data_npz,numero_face):

    for item in numero_face:
        imagem = load_jpg(data_npz,item)

        face_landmarks_list = face_recognition.face_landmarks(imagem)

        print("Rosto {} : I found {} face(s) in this photograph.".format(item,len(face_landmarks_list)))
        
        # Create a PIL imagedraw object so we can draw on the picture
        pil_image = Image.fromarray(imagem)
        d = ImageDraw.Draw(pil_image)


        mediaX = mediaY = somaX = somaY = float()
        qts_points = int()


        for face_landmarks in face_landmarks_list:
            qts_points = len(face_landmarks['left_eye'])
            raio = abs(((face_landmarks['left_eye'][1][1]) -  (face_landmarks['left_eye'][4][1]))* 0.6)
            # print(raio)
            for point in face_landmarks['left_eye']:
                somaX += point[0]
                somaY += point[1] 
            
            mediaX=somaX/qts_points
            mediaY=somaY/qts_points


            d.ellipse((mediaX-raio,mediaY-raio,mediaX+raio,mediaY+raio), outline ="blue",width=2)
            d.point((mediaX, mediaY), fill="yellow")

            print("Olho Esquerdo x = {} , y = {} ".format(mediaX,mediaY))
            print("Olho Esquerdo CSV : X = {} Y = {}".format(data_csv['right_eye_center_x'][item],data_csv['right_eye_center_y'][item]))
            print('\nErro Olho Esquerdo: {} '.format(calc_erro(mediaX,mediaY,data_csv['right_eye_center_x'][item],data_csv['right_eye_center_y'][item])))

            print('--------------------------------------')
            somaX = somaY = 0

            for point in face_landmarks['right_eye']:
                somaX += point[0]
                somaY += point[1] 
                # print(point,somaX,somaY)
            
            mediaX=somaX/qts_points
            mediaY=somaY/qts_points

            d.ellipse((mediaX-raio,mediaY-raio,mediaX+raio,mediaY+raio), outline ="blue",width=2)
            d.point((mediaX, mediaY), fill="yellow")

            print("Olho Direito x = {} , y = {} ".format(mediaX,mediaY))
            print("Olho Direito CSV : X = {} Y = {}".format(data_csv['left_eye_center_x'][item],data_csv['left_eye_center_y'][item]))
            print('\nErro Olho Direito: {} '.format(calc_erro(mediaX,mediaY,data_csv['left_eye_center_x'][item],data_csv['left_eye_center_y'][item])))

            # Print the location of each facial feature in this image
            # for facial_feature in face_landmarks.keys():
            # 	if facial_feature == "left_eye" or facial_feature == "right_eye":
            # 		print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

                    
            # Let's trace out each facial feature in the image with a line!
            for facial_feature in face_landmarks.keys():
                if facial_feature == "left_eye" or facial_feature == "right_eye":
                    d.line(face_landmarks[facial_feature], width=2,fill="red")

        # print(face_data['left_eye_center_x'][k])

        # Show the picture
        pil_image.show()

        
        
        print('===============================================================================')
        



# Lê arquivo .CSV com os dados dos rostos
face_data =pd.read_csv('facial_keypoints.csv')

# Lê arquivo .npz com as imagens
imagens = read_npz('face_images.npz')

#sorteia  uma quantidade de rosto 
rostos = random_faces(imagens,3)

main_program(face_data,imagens,rostos)






