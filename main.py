from fastapi import FastAPI, UploadFile, File
import cv2
import pytesseract
import re
import numpy as np
from typing import List

app = FastAPI()

def procesar_imagen(imagen_gris):
    # Preprocesamiento de la imagen (por ejemplo, ajuste de contraste, binarización, eliminación de ruido)
    # Binarizar la imagen utilizando un umbral
    umbral, imagen_binarizada = cv2.threshold(imagen_gris, 245, 255, cv2.THRESH_BINARY)

    # Detección de contornos
    contornos, _ = cv2.findContours(imagen_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prediccion = list()

    # Itera sobre los contornos y extrae subimágenes
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        # Asegúrate de que el contorno tenga un área mínima para filtrar pequeños detalles
        if cv2.contourArea(contorno) > 100:
            # Recorta la subimagen, agrega un pequeño margen para evitar el borde del dígito
            margen = 12
            subimagen = imagen_binarizada[y + margen:y + h - margen, x + margen:x + w - margen]

            # Aplica OCR a la subimagen
            texto = pytesseract.image_to_string(subimagen, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')  # 'psm 6' para tratar de reconocer un solo dígito
            x = re.search("\d", texto)
            prediccion.append(int(x.group()))

    prediccion_1 = np.array(prediccion).reshape(5,5)
    prediccion_1 = np.flip(prediccion_1)

    clave = np.concatenate((prediccion_1[:,0], prediccion_1[4,1:]), axis=0)
    clave_list_str = list(map(lambda x: str(x),a))
    return  ''.join(clave_list_str)



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Procesar la imagen
    resultado = procesar_imagen(img_np)

    return {"prediccion": resultado}
