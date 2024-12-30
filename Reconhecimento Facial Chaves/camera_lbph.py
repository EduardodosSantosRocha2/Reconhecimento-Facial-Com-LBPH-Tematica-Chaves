import cv2
detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')
reconhecedor_face = cv2.face.LBPHFaceRecognizer_create()
reconhecedor_face.read("lbph_classifier.yml")
altura, largura = 220, 220

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector_face.detectMultiScale(imagem_cinza, minSize=(30, 30), scaleFactor=1.5)

    for x, y, w, h in deteccoes:
        imagem_face = cv2.resize(imagem_cinza[y: y+w, x: x+h], (largura, altura))
        cv2.rectangle(frame, (x,y), (x + w , y+ h),(0, 255, 0), 2)
        id, confidence = reconhecedor_face.predict(imagem_face)
        nome = ""
        if id == 1:
            nome = "Silvio Santos"
        elif id == 2:
            nome = "Chavinho"
        elif id == 3:
            nome = "Quico"
         
        cv2.putText(frame, nome, (x, y + (h+30)), font ,2, (0, 255,0))
        # cv2.putText(frame, str(confidence), (x, y + (h+50)), font,1, (0, 255,0))
    
    #Mostrar o resultado    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Libera a memoria
camera.release()
cv2.destroyAllWindows()