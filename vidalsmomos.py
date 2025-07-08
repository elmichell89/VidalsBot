#Vidalsmomos Bot - Prueba
#verde #00ff00 - HSV 120,100,100
#Falta:
# - añadir función para tomar los archivos de manera aleatoria dentro de una carpeta --> checar que jalen lineas 9,10,12,13
# - entender bien qué hace a partir de la línea 29

import cv2 as cv
import numpy as np
#import os
#import random

#template = cv.imread(random.choice(os.listdir("")))
#source = cv.imread(random.choice(os.listdir("")))

template = cv.imread('template1.jpg')
source = cv.imread('source1.jpg')

#convertir imagen de BGR a HSV
template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
#rango de color
lower_g = np.array([55, 250, 250])
upper_g = np.array([65, 255, 255])

#mascara del verde
mask = cv.inRange(template_hsv, lower_g, upper_g)
#mascara de lo que no es verde
mask_inv = cv.bitwise_not(mask)

#busca las coordenadas del verde
coords = cv.findNonZero(mask)
#obtiene un rectangulo del area donde se encuentra el verde
x,y,w,h = cv.boundingRect(coords)

#obtiene el tamaño original del source
src_h, src_w = source.shape[:2]
#calcula el factor de escala para que el source no se deforme dentro del verde
scale = min(w / src_w, h / src_h)
#calcula nuevas medidas del source reescalado
new_w = int(src_w * scale)
new_h = int(src_h * scale)

#redimensiona el source
source_resized = cv.resize(source, (new_w, new_h), interpolation = cv.INTER_AREA)

#rellena el area verde de negro
source_padded = np.zeros((h,w,3), dtype=np.uint8)

#calcula el offset para centrar la imagen redimensionada dentro del verde
offset_x = (w - new_w) // 2
offset_y = (h - new_h) // 2
#coloca el source reescalado al centro del area verde
source_padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = source_resized

#ROI = region of interest = area verde, la extrae
roi = template[y:y+h, x:x+w]
#obtiene las mascaras para el area verde
mask_crop = mask[y:y+h, x:x+w]
mask_crop_inv = cv.bitwise_not(mask_crop)

#elimina el color verde del template original
template_bg = cv.bitwise_and(roi, roi, mask=mask_crop_inv)

#aplica la mascara al source
source_fg = cv.bitwise_and(source_padded, source_padded, mask=mask_crop)

#combina template y source
dst = cv.add(template_bg, source_fg)

#inserta la region editada en la plantilla
imagen_final = template.copy()
imagen_final[y:y+h, x:x+w] = dst

cv.imshow('original', template)
cv.imshow('editada', imagen_final)
cv.imwrite('vidalsmomo.jpg', imagen_final)
cv.waitKey(0)
cv.destroyAllWindows()
