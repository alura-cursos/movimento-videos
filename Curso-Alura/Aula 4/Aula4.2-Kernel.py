import numpy as np 
import cv2
import sys 

VIDEO = 'Dados/Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithm_types[1]


def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

print("Dilation: ")
print(Kernel('dilation'))

print("Opening: ")
print(Kernel('opening'))

print("Closing: ")
print(Kernel('closing'))

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
background_subtractor = Subtractor(algorithm_type)


def main():
    while (cap.isOpened):
        ok, frame = cap.read()

        if not ok:
          print('Frames acabaram!')
          break
      
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


        mask = background_subtractor.apply(frame)
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            break
        
    
main()