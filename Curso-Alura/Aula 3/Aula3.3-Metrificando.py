import cv2
import sys

VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithm_types[3]

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
    print('Erro')
    sys.exit(1)
    

cap = cv2.VideoCapture(VIDEO)
background_subtractor = Subtractor(algorithm_type)


def main():
    while (cap.isOpened):
        ok, frame = cap.read()

        if not ok:
          print('Frames acabaram!')
          break

        mask = background_subtractor.apply(frame)
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            break
main()