#Vidals Bot 1.0
#verde #00ff00 - HSV 120,100,100

#Posibles Mejoras:
# - Colocar path específico para guardar las imagenes resultantes

import cv2 as cv
import numpy as np
import os
import random

#os.listdir solamente da el nombre de los archivos, no las rutas completas
#y opencv solamente intenta leer 'templateX.jpg' sin saber en que carpeta está
#se tiene que unir la ruta con el nombre del archivo usando el os.path.join

#rutas de las carpetas
template_folder = 'vidalsmomosbot-main/templates/'
source_folder = 'vidalsmomosbot-main/sources/'
#une las rutas guardadas arriba con el nombre de un archivo random
template_path = os.path.join(template_folder, random.choice(os.listdir(template_folder)))
source_path = os.path.join(source_folder, random.choice(os.listdir(source_folder)))

template = cv.imread(template_path)
source = cv.imread(source_path)

#template = cv.imread('vidalsmomosbot-main/template2.jpg')
#source = cv.imread('vidalsmomosbot-main/source2.jpg')

#convertir imagen de BGR a HSV
template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
#rango de color para evitar pedos
lower_g = np.array([55, 250, 250])
upper_g = np.array([65, 255, 255])

#hace una mascara del verde
mask = cv.inRange(template_hsv, lower_g, upper_g)
#mascara de lo que no es verde a.k.a. invierte la mascara anterior
mask_inv = cv.bitwise_not(mask)

#busca las coordenadas del verde
coords = cv.findNonZero(mask)
#obtiene un rectangulo del area donde se encuentra el verde
x,y,w,h = cv.boundingRect(coords)

#Cambiar el tamaño de la imagen source:
#obtiene el tamaño original del source
src_h, src_w = source.shape[:2]
#calcula el factor de escala para que el source no se deforme dentro del verde
scale = min(w / src_w, h / src_h)
#calcula nuevas medidas del source reescalado
new_w = int(src_w * scale)
new_h = int(src_h * scale)
#redimensiona el source
source_resized = cv.resize(source, (new_w, new_h), interpolation = cv.INTER_AREA)

#hace una imagen negra (blank) para después sobreponerla al verde sobrante de la template
relleno = np.zeros((h,w,3), dtype=np.uint8)

#calcula el offset para centrar la imagen source redimensionada dentro del verde
offset_x = (w - new_w) // 2
offset_y = (h - new_h) // 2
#coloca el source reescalado sobre la imagen negra de relleno y posteriormente al centro del area verde
relleno[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = source_resized

#aisla el area verde
a_verde = template[y:y+h, x:x+w]
#obtiene las mascaras para el area verde
mask_crop = mask[y:y+h, x:x+w]
mask_crop_inv = cv.bitwise_not(mask_crop)

#elimina el color verde
template_bg = cv.bitwise_and(a_verde, a_verde, mask=mask_crop_inv)

#aplica la mascara del area verde al source
source_fg = cv.bitwise_and(relleno, relleno, mask=mask_crop)

#combina la máscara anterior con el source para después unirla al template
img_editada = cv.add(template_bg, source_fg)

#inserta la region editada en la plantilla
imagen_final = template.copy() #crea una copia de la template original
imagen_final[y:y+h, x:x+w] = img_editada #coloca la img_editada sobre la región verde de la imagen de arriba

cv.imwrite('vidalsmomo.jpg', imagen_final)
cv.waitKey(0)
cv.destroyAllWindows()
