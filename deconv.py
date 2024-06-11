import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import grad_x,grad_y,norma_p
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
mascara = np.zeros_like(imagen[:, :, :])
lam = 10
mascara[:,:,:] = lam

altura, anchura = imagen.shape[0], imagen.shape[1]
num_circulos, radio_min, radio_max = 20,5,12
puntos = np.random.randint(0, min(anchura, altura), size=(num_circulos, 2))
radios = np.random.randint(radio_min, radio_max, size=num_circulos)

for punto, radio in zip(puntos, radios):
        cv2.circle(mascara, tuple(punto), radio, (0,0,0), thickness=-1)  # thickness=-1 llena el círculo
        cv2.circle(imagen, tuple(punto), radio, (0,0,0), thickness=-1)  # thickness=-1 llena el círculo
    

#mascara[y1:y2, x1:x2,:] = 0
#imagen[y1:y2, x1:x2,:] = 0



dt = 0.1
f = imagen.copy()

u = imagen.copy()
for _ in range(1,7000):
  #Algoritmo
  u_x = grad_x(u)
  u_y = grad_y(u)

  normap = norma_p(u,2)
  lap = grad_x(normap * u_x) + grad_y(normap * u_y)
    
  wk = -lap + mascara*(u-f)
    
  u = u - dt*wk


# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Mostrar la imagen original en el primer subgráfico
axes[0].imshow(original_image)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

# Mostrar la imagen con ruido en el segundo subgráfico
axes[1].imshow(imagen)
axes[1].set_title('Máscara')
axes[1].axis('off')

u = u / 255
#u = u - imagen
axes[2].imshow(u)
axes[2].set_title('Inpainting')
axes[2].axis('off')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Guardar la figura en un archivo
plt.savefig('inpainting.png')

