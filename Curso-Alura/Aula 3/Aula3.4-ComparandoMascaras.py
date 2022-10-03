import cv2
import sys


VIDEO = 'Dados/Ponte.mp4'


algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Detector inválido')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO)

background_subtractor = []

for i, a in enumerate(algorithm_types): # gera um ID para cada um dos algoritmos
    #print(i, a)
    background_subtractor.append(Subtractor(a))

def main():
    frame_number = 0
    while (cap.isOpened):
        
        ok, frame = cap.read()

        if not ok:
            print('Erro na captura')
            break

        frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

        gmg = background_subtractor[0].apply(frame)
        mog = background_subtractor[1].apply(frame)
        mog2 = background_subtractor[2].apply(frame)
        knn = background_subtractor[3].apply(frame)
        cnt = background_subtractor[4].apply(frame)

    
        cv2.imshow('Original', frame)
        cv2.imshow('GMG', gmg)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)


        k = cv2.waitKey(0) & 0xff
        if k == 27: # ESC
            break

main()