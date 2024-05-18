import os
import cv2
import numpy as np
from cca import CCA_2D
from pls import PLS_2D

def process_images(input_dir, image_num = None):
    X_full = []
    if image_num is None:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_dir, filename)
                #print(image_path)
                image = cv2.imread(image_path)
                # Convert image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Append the grayscale image to X_full without flattening
                X_full.append(gray_image)
                
        # Convert the list of grayscale images to a numpy array
        X_arrays = np.array(X_full, dtype=np.uint8)
        return X_arrays
    else:
        image_paths = [filename for filename in os.listdir(input_dir) if filename.endswith(".jpg") or filename.endswith(".png")]
        if image_num >= len(image_paths):
            raise ValueError("Invalid image_num. There are only {} images available.".format(len(image_paths)))
        
        image_path = os.path.join(input_dir, image_paths[image_num])
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
def process(num_of_picture = 1, accuracy_coef = 0.1):
    input_dirX = os.path.join(os.path.dirname(__file__), "input", "X")
    input_dirY = os.path.join(os.path.dirname(__file__), "input", "Y")
    output_dirX_cca = os.path.join(os.path.dirname(__file__), "outputcca", "X")
    output_dirY_cca = os.path.join(os.path.dirname(__file__), "outputcca", "Y")
    output_dirX_pls = os.path.join(os.path.dirname(__file__), "outputpls", "X")
    output_dirY_pls = os.path.join(os.path.dirname(__file__), "outputpls", "Y")
    current_dir = os.path.dirname(__file__)
    input_dirG = os.path.join(current_dir, "clear", "dogs")
    input_dirL = os.path.join(current_dir, "clear", "humans")
    K,x,y = process_images(input_dirX).shape
    print("Number of loaded images")
    print(K)
    print("Size of image", x,"x",y)
    X = process_images(input_dirX)
    Y = process_images(input_dirY)
    image_num = num_of_picture
    G = process_images(input_dirG, image_num - 1)
    L = process_images(input_dirL, image_num - 1)
    G = np.array(G, dtype=np.uint8)
    cv2.imwrite('input_image1.jpg', G)
    cv2.imwrite('input_image2.jpg', L)
    result_cca1, result_cca2, reconstructed_x_cca, reconstructed_y_cca = CCA_2D(K, X, Y, G, L, accuracy_coef = accuracy_coef)
    result_pls1, result_pls2, reconstructed_x_pls, reconstructed_y_pls = PLS_2D(K, X, Y, G, L, accuracy_coef = accuracy_coef)


    if result_cca1 is not None:
        image_array_cca1 = np.array(result_cca1, dtype=np.float64)
        cv2.imwrite('output_cca1.jpg', image_array_cca1)
    if result_pls1 is not None:
        image_array_pls1 = np.array(result_pls1, dtype=np.float64)
        image_array_pls1 = 255 - image_array_pls1
        cv2.imwrite('output_pls1.jpg', image_array_pls1)
    if result_cca2 is not None:
        image_array_cca2 = np.array(result_cca2, dtype=np.float64)
        cv2.imwrite('output_cca2.jpg', image_array_cca2)
    if result_pls2 is not None:
        image_array_pls2 = np.array(result_pls2, dtype=np.float64)
        image_array_pls2 = 255 - image_array_pls2
        cv2.imwrite('output_pls2.jpg', image_array_pls2)
    for i in range(0,K):
        path_X = os.path.join(output_dirX_cca, str(i) + ".jpg")
        cv2.imwrite(path_X, reconstructed_x_cca[i])
        path_Y = os.path.join(output_dirY_cca, str(i) + ".jpg")
        cv2.imwrite(path_Y, reconstructed_y_cca[i])
    for i in range(0,K):
        path_X = os.path.join(output_dirX_pls, str(i) + ".jpg")
        cv2.imwrite(path_X, reconstructed_x_pls[i])
        path_Y = os.path.join(output_dirY_pls, str(i) + ".jpg")
        cv2.imwrite(path_Y, reconstructed_y_pls[i])

    print("Finished. Open folder to see results")
if __name__ == "__main__":
    process(1)
