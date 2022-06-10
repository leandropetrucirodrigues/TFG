from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pandas as pd
import random

face_data = pd.read_csv('facial_keypoints.csv')
# na_idx = set()
# for col in face_data.columns:
#     for idx in face_data[face_data[col].isnull()].index.tolist():
#         na_idx.add(idx)

# na_idx = list(na_idx)
# face_data = face_data.dropna().reset_index(drop = True)


image_data = np.load('face_images.npz')
images = image_data['face_images']
images = np.transpose(images,(2,0,1))
imagem = []
for i,ima in enumerate(images):
    imagem.append(ima)

imagem = np.array(imagem)
# imagem = np.expand_dims(imagem,axis =-1)
imagem = np.asarray(imagem)
# print(len(imagem))
qts_rosto = random.sample(range(0,len(imagem)),8)
# print(qts_rosto)
for k in qts_rosto:

# Load the jpg file into a numpy array
	image2 = Image.fromarray(imagem[k])
	imeg = image2.convert("RGB")
	imeg = np.array(imeg)
	

	# Find all facial features in all the faces in the image
	face_landmarks_list = face_recognition.face_landmarks(imeg)

	print("Rosto {} : I found {} face(s) in this photograph.".format(k,len(face_landmarks_list)))

	# Create a PIL imagedraw object so we can draw on the picture
	pil_image = Image.fromarray(imeg)
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
			# print(point,somaX,somaY)
		
		mediaX=somaX/qts_points
		mediaY=somaY/qts_points

		d.ellipse((mediaX-raio,mediaY-raio,mediaX+raio,mediaY+raio), outline ="blue",width=2)
		d.point((mediaX, mediaY), fill="yellow")

		print("Olho Esquerdo x = {} , y = {} ".format(mediaX,mediaY))
		print("Olho Esquerdo CSV : X = {} Y = {}".format(face_data['right_eye_center_x'][k],face_data['right_eye_center_y'][k]))

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
		print("Olho Direito CSV : X = {} Y = {}".format(face_data['left_eye_center_x'][k],face_data['left_eye_center_y'][k]))

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

	
	
	print('----------------------------------------------------------------------------')