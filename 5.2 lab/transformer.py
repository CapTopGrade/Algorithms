from PIL import Image
import os 
import numpy as np
import cv2

file_path = os.path.dirname(__file__)

def find_white_line(image):
    # Получаем размеры изображения
    width, height = image.size
    
    # Находим позицию белой линии
    white_line_position = None
    for x in range(width):
        pixels = [image.getpixel((x, y)) for y in range(height)]
        if all(pixel == (255, 255, 255) for pixel in pixels):  # Если весь столбец белый
            white_line_position = x
            break
    
    return white_line_position

def split_image(image_path):
    # Открываем изображение
    image = Image.open(image_path)
    
    # Получаем размеры изображения
    width, height = image.size
    
    left_image = image.crop((15, 0, 440-15, height))
    right_image = image.crop((450 + 15, 0, width-15, height))
    return left_image, right_image

def add_gaussian_noise(image, amplitude):
    noise = np.random.normal(0, amplitude, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def bw_process_images(image):
    image = image.resize((410,650))
    image = np.array(image)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Save the processed images
    amplitude = 0.05 * np.max(bw_image)
    bw_image = add_gaussian_noise(bw_image, amplitude)
    return bw_image
            

# Пример использования функции
for i in range(1,27):
    left_image, right_image = split_image(os.path.join(file_path, "clear", str(i)+".jpg"))
    left_image = bw_process_images(left_image)
    right_image = bw_process_images(right_image)
    if i < 10:
        cv2.imwrite(os.path.join(file_path,"clear", "dogs", "0" + str(i) + ".jpg"), left_image)
        cv2.imwrite(os.path.join(file_path,"clear", "humans",  "0" + str(i) + ".jpg"), right_image)
    else:    
        cv2.imwrite(os.path.join(file_path,"clear", "dogs", str(i) + ".jpg"), left_image)
        cv2.imwrite(os.path.join(file_path,"clear", "humans",  str(i) + ".jpg"), right_image)
    
