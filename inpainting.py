import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import grad_x,grad_y,lap
import random
# Cargar la imagen
image_path = "images/rock_transgresivo.jpg"  # Ruta de la imagen
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RG
imagen = original_image.copy()

# Generar coordenadas aleatorias para el cuadrado
x1 = random.randint(50, 200)
y1 = random.randint(50, 200)
x2 = random.randint(x1, 300)
y2 = random.randint(y1, 400)

# Crear una máscara con el cuadrado
mascara = np.zeros_like(imagen[:, :, 0])
mascara[y1:y2, x1:x2] = 255
# Dibujar el cuadrado relleno sobre la imagen
color = (0, 0, 0)  # Color negro
cv2.rectangle(imagen, (x1, y1), (x2, y2), color, -1)  # Grosor -1 para rellenar





# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Mostrar la imagen original en el primer subgráfico
axes[0].imshow(original_image)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

# Mostrar la imagen con ruido en el segundo subgráfico
axes[1].imshow(imagen)
axes[1].set_title('Imagen con Ruido Gaussiano')
axes[1].axis('off')



# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Guardar la figura en un archivo
plt.savefig('inpainting.png')
