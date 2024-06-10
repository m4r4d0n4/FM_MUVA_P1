import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import grad_x,grad_y,lap

def add_gaussian_noise(image, mean=0, sigma=45):
    """
    Function to add Gaussian noise to an image.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Cargar la imagen
image_path = "images/rock_transgresivo.jpg"  # Ruta de la imagen
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB

# Aplicar ruido gaussiano a la imagen
noisy_image = add_gaussian_noise(original_image)

u = noisy_image.copy()
lam = 0.5
dt = 0.001

for _ in range(1,700):

    lap1 = lap(u)
    w_k = -lap1 + lam * (u - noisy_image)
    u = u - dt * w_k


# Crear una figura con dos subgráficos
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Mostrar la imagen original en el primer subgráfico
axes[0].imshow(original_image)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

# Mostrar la imagen con ruido en el segundo subgráfico
axes[1].imshow(noisy_image)
axes[1].set_title('Imagen con Ruido Gaussiano')
axes[1].axis('off')

u = u / 255
axes[2].imshow(u)
axes[2].set_title('Imagen Denoised')
axes[2].axis('off')


# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Guardar la figura en un archivo
plt.savefig('original_vs_ruido.png')




