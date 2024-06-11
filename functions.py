import numpy as np

def grad_x(image):
    """
    Calcula la derivada en la dirección x de una imagen utilizando diferencias finitas centradas de primer orden.
    """
    # Calcular la derivada en la dirección x utilizando diferencias finitas centradas de primer orden
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_x[:, 1:-1,:] = (image[:, 2:,:] - image[:, :-2,:])/2
    grad_x[:, 0] = (image[:, 1,:] - image[:, 0,:])
    grad_x[:, -1] = (image[:, -1,:] - image[:, -2,:])
    return grad_x

def grad_y(image):
    """
    Calcula la derivada en la dirección y de una imagen utilizando diferencias finitas centradas de primer orden.
    """
    # Calcular la derivada en la dirección y utilizando diferencias finitas centradas de primer orden
    grad_y = np.zeros_like(image, dtype=np.float64)
    grad_y[1:-1, :] = (image[2:, :,:] - image[:-2, :,:])/2
    grad_y[0, :] = (image[1, :,:] - image[0, :,:])
    grad_y[-1, :] = (image[-1, :,:] - image[-2, :,:])
    return grad_y

def lap(image):

    gradx = grad_x(image)
    grady = grad_y(image)
    
    return grad_x(gradx) + grad_y(grady)

def norma_p(imagen,p,epsilon=0.001):
    
    norma = np.sqrt(grad_x(imagen)**2 + grad_y(imagen)**2 + epsilon**2)
    return norma**(p-2)
