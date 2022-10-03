import numpy as np
import cv2
import sys

VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

#-------------------------------------------------------------------------------------------------------------------------

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation
    
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
    
    
#-------------------------------------------------------------------------------------------------------------------------

w_min = 30  # largura minima do retangulo
h_min = 30  # altura minima do retangulo
offset = 10  # Erro permitido entre pixel
linha_ROI = 500  # Posição da linha de contagem
carros = 0



def centroide(x, y, w, h):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param w: largura do objeto
    :param h: altura do objeto
    :return: tupla que contém as coordenadas do centro de um objeto
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


detec = []
def set_info(detec):
    global carros
    for (x, y) in detec:
        if (linha_ROI + offset) > y > (linha_ROI - offset):
            carros += 1
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))


def show_info(frame, mask):
    text = f'Carros: {carros}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", mask)



cap = cv2.VideoCapture(VIDEO) #lendo o vídeo

fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec do vídeo

algorithm_type = algorithm_types[1]
background_subtractor = Subtractor(algorithm_type)  # Pega o fundo e subtrai do que está se movendo

while True:
    
    ok, frame = cap.read() # Pega cada frame do vídeo
    
    if not ok:
        break
    
    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')
    
    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = centroide(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame, mask)
    
    
    if cv2.waitKey(1) == 27: #ESC
        break

cv2.destroyAllWindows()
cap.release()