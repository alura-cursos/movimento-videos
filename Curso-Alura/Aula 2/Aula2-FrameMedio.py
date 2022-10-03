## Aula 1 ------------------
import numpy as np
from time import sleep
import cv2

delay = 10
VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Rua.mp4'
VIDEO_OUT = 'Dados/Resultados/filtragem.avi'

cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 72, (frame.shape[1], frame.shape[0]), False)


framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)


medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
#print(medianFrame)
#cv2.imshow('Median frame', medianFrame)
#cv2.waitKey(0)
cv2.imwrite('median_frame-pisa.jpg', medianFrame)


## Pode ser um desafio isso aqui: Mostrar todos os 72 frames
#for frame in frames:
   #cv2.imshow('Frame', frame)
   #cv2.waitKey(0)


## Aula 2 --------------------------

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', grayMedianFrame)
cv2.waitKey(0)

while (True):
    tempo = float(1 / delay)
    sleep(tempo)
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Acabou os frames')
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)

    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('frame', dframe)
    writer.write(dframe)
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

writer.release()
cap.release()
