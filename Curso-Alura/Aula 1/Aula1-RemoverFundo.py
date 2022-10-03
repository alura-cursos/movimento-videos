import numpy as np
import cv2

#VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Arco Do Triunfo - 17.mp4'
#VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Estradas.mp4'
#VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/peixinhos.mp4'
VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Rua.mp4'

cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)
    
    
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
print(medianFrame)
cv2.imshow('Median frame', medianFrame)
cv2.waitKey(0)
cv2.imwrite('median_frame.jpg', medianFrame)