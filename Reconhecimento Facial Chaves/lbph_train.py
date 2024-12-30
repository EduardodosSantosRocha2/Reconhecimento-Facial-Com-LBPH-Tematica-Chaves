import cv2
import os
from PIL import Image
import numpy as np

def get_image_data():

  paths = [os.path.join('./yalefaces/train', f) for f in os.listdir('./yalefaces/train')]
  faces = []
  ids = []
  print(paths)
  for path in paths:
    imagem = Image.open(path).convert('L')

    imagem_np = np.array(imagem, 'uint8')
    faces.append(imagem_np)

    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    ids.append(id)

  return np.array(ids), faces



ids, faces = get_image_data()
lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius = 4, neighbors = 14, grid_x = 10, grid_y = 10) #radiu  = alcance, Neighbors = numero de total de pixels vizinhos, grid_x e grid_y = tamanho da grade para gerar os histogramas, Threshold = indica a confiança
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')
