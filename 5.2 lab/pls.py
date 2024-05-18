import numpy as np
import json
from joblib import Parallel, delayed
import time
import warnings
import matplotlib.pyplot as plt
import numpy.linalg as la

# Filter out the specific warning
warnings.filterwarnings("ignore", category=np.ComplexWarning)

K = 4 #number of vectors
np.set_printoptions(suppress=True, threshold=1e-12)

def find_closest_vector1(U_s, V,  K = 2, accuracy_coef = 1):
    min_distance = np.inf
    closest_vector = None
    
    for i in range(0,K):
        distance = np.max(np.abs(U_s - V[i])) * accuracy_coef
        if distance < min_distance:
            min_distance = distance
            closest_vector = -V[i]
    
    return closest_vector, min_distance

def find_closest_vector2(V_s, U,  K = 2, accuracy_coef = 1):
    min_distance = np.inf
    closest_vector = None
    
    for i in range(0,K):
        distance = np.max(np.abs(V_s - U[i])) * accuracy_coef
        if distance < min_distance:
            min_distance = distance
            closest_vector = -U[i]
    return closest_vector, min_distance

def sort_eigenvalues_and_vectors(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

def train(X, Y, K = K, accuracy_coef = 1):
    def mean(Matrix):
        A_num, _ , _= Matrix.shape
        A_sum = np.sum(Matrix, axis=0)
        A_mean = (1 / A_num) * A_sum
        return A_mean
    
    def centrate(matrixes, counted_mean, K = 2):
        centrated_matrix = matrixes 
        centrated_matrix = np.array(centrated_matrix ,dtype=np.float64)
        for i in range(0,K):
            centrated_matrix[i] = matrixes[i] - counted_mean
        return centrated_matrix
    
    def calculate_covariate_matrices(centrated_matrix_x, centrated_matrix_y, i):
        centrated_matrix_x_T = centrated_matrix_x[i].T
        centrated_matrix_y_T = centrated_matrix_y[i].T

        C_xy_r = centrated_matrix_x[i] @ centrated_matrix_y_T
        C_yx_r = C_xy_r.T

        C_xy_c = centrated_matrix_x_T @ centrated_matrix_y[i]
        C_yx_c = C_xy_c.T 

        return C_xy_r, C_yx_r, C_xy_c, C_yx_c
            
    def covariate_count_eig_weight_parallel(centrated_matrix_x, centrated_matrix_y, K=2):
        results = Parallel(n_jobs=-1)(delayed(calculate_covariate_matrices)(centrated_matrix_x, centrated_matrix_y, i) for i in range(K))
        start = time.time()
        
        sum_C_xy_r = sum(result[0] for result in results)
        sum_C_yx_r = sum(result[1] for result in results)
        sum_C_xy_c = sum(result[2] for result in results)
        sum_C_yx_c = sum(result[3] for result in results)

        S_1_r = sum_C_xy_r @ sum_C_yx_r
        S_2_r = sum_C_yx_r @ sum_C_xy_r

        S_1_c = sum_C_xy_c @ sum_C_yx_c
        S_2_c = sum_C_yx_c @ sum_C_xy_c

        delta_x1, W_x1 = np.linalg.eig(S_1_r)
        delta_y1, W_y1 = np.linalg.eig(S_2_r)

        delta_x1, W_x1 = sort_eigenvalues_and_vectors(delta_x1, W_x1)
        delta_y1, W_y1 = sort_eigenvalues_and_vectors(delta_y1, W_y1)
        
        delta_x2, W_x2 = np.linalg.eig(S_1_c)
        delta_y2, W_y2 = np.linalg.eig(S_2_c)
        
        delta_x2, W_x2 = sort_eigenvalues_and_vectors(delta_x2, W_x2)
        delta_y2, W_y2 = sort_eigenvalues_and_vectors(delta_y2, W_y2)  
        
        U = centrated_matrix_x
        V = centrated_matrix_y
        
        for i in range(0,K):
            U[i] = W_x1.T @ centrated_matrix_x[i] @ W_x2
            V[i] = W_y1.T @ centrated_matrix_y[i] @ W_y2
            
        end = time.time()
        time_spend = end-start
        
        # correlation calculating
        u_corrs, v_corrs = [], []
        for i in range(len(U)):
            u_corrs.append(la.norm(U[i]) / la.norm(U))
            v_corrs.append(la.norm(V[i]) / la.norm(V))

        # show correlation
        plt.plot(u_corrs, label='U', color='blue')
        plt.plot(v_corrs, label='V', color='orange')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('Index')
        plt.ylabel('Correlation')
        plt.title('Correlation Plot 2DPLS Parallel')
        plt.show()

        # show eigenvalues
        fig, ax = plt.subplots(figsize=(10, 8))
        bins = 50
        hist_x1, bins_x1 = np.histogram(delta_x1, bins=bins)
        hist_x2, bins_x2 = np.histogram(delta_x2, bins=bins)

        for i, (val1, val2) in enumerate(zip(hist_x1, hist_x2)):
            if val1 != 0:
                ax.scatter(bins_x1[i], val1, marker='o', s=100, color='blue')
                ax.vlines(bins_x1[i], 0, val1, linestyle='--', color='blue', alpha=0.5)
            if val2 != 0:
                ax.scatter(bins_x2[i], val2, marker='^', s=100, color='orange')
                ax.vlines(bins_x2[i], 0, val2, linestyle='--', color='orange', alpha=0.5)
                
        ax.scatter([], [], label='Delta_1', marker='o', s=100, color='blue')
        ax.scatter([], [], label='Delta_2', marker='^', s=100, color='orange')
        ax.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Delta Values Frequency 2DPLS Parallel')
        plt.show()
            
        return delta_x1, delta_y1, delta_x2, delta_y2, W_x1, W_y1, W_x2, W_y2, U, V, time_spend
    
    def covariate_count_eig_weight_cascade(centrated_matrix_x, centrated_matrix_y, K = 2):
        start = time.time()
        for i in range(0,K):
            _,x,y = centrated_matrix_x.shape
            _,j,k = centrated_matrix_x.shape
            z = x*y
            h = j*k
            centrated_matrix_x_T = centrated_matrix_x[i]
            centrated_matrix_y_T = centrated_matrix_y[i]
            centrated_matrix_x_T = centrated_matrix_x_T.T
            centrated_matrix_y_T = centrated_matrix_y_T.T
            
            C_xy_r = (1/z)*(centrated_matrix_x[i] @ centrated_matrix_y_T)
            C_yx_r = C_xy_r.T
            
            C_xy_c = centrated_matrix_x_T @ centrated_matrix_y[i]
            C_yx_c = C_xy_c.T 

            if i == 0:
                sum_C_xy_r = np.zeros_like(C_xy_r)
                sum_C_yx_r = np.zeros_like(C_yx_r)
                sum_C_xy_c = np.zeros_like(C_xy_c)
                sum_C_yx_c = np.zeros_like(C_yx_c)
            
            sum_C_xy_r += C_xy_r
            sum_C_yx_r += C_yx_r
            sum_C_xy_c += C_xy_c
            sum_C_yx_c += C_yx_c
                        
        S_1_r = sum_C_xy_r @ sum_C_yx_r
        S_2_r = sum_C_yx_r @ sum_C_xy_r
        
        delta_x1, W_x1 = np.linalg.eig(S_1_r)
        delta_y1, W_y1 = np.linalg.eig(S_2_r)
        
        delta_x1, W_x1 = sort_eigenvalues_and_vectors(delta_x1, W_x1)
        delta_y1, W_y1 = sort_eigenvalues_and_vectors(delta_y1, W_y1)
        
        U_1 = np.zeros_like(centrated_matrix_x)
        V_1 = np.zeros_like(centrated_matrix_y)
        for j in range(0,K):
            U_1[j] = W_x1.T @ centrated_matrix_x[j]
            V_1[j] = W_y1.T @ centrated_matrix_y[j]
            
        
        S_1_c = sum_C_xy_c @ sum_C_yx_c
        S_2_c = sum_C_yx_c @ sum_C_xy_c
        
        delta_x2, W_x2 = np.linalg.eig(S_1_c)
        delta_y2, W_y2 = np.linalg.eig(S_2_c)
                    
        delta_x2, W_x2 = sort_eigenvalues_and_vectors(delta_x2, W_x2)
        delta_y2, W_y2 = sort_eigenvalues_and_vectors(delta_y2, W_y2)
        
        U = np.zeros_like(centrated_matrix_x)
        V = np.zeros_like(centrated_matrix_y)
        for j in range(0,K):
            U[j] = U_1[j] @ W_x2
            V[j] = V_1[j] @ W_y2
        end = time.time()
        time_spend = end - start 
                # correlation calculating
        u_corrs, v_corrs = [], []
        for i in range(len(U)):
            u_corrs.append(la.norm(U[i]) / la.norm(U))
            v_corrs.append(la.norm(V[i]) / la.norm(V))

        # show correlation
        plt.plot(u_corrs, label='U', color='blue')
        plt.plot(v_corrs, label='V', color='orange')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('Index')
        plt.ylabel('Correlation')
        plt.title('Correlation Plot 2DPLS Cascade')
        plt.show()

        # show eigenvalues
        fig, ax = plt.subplots(figsize=(10, 8))
        bins = 50
        hist_x1, bins_x1 = np.histogram(delta_x1, bins=bins)
        hist_x2, bins_x2 = np.histogram(delta_x2, bins=bins)

        for i, (val1, val2) in enumerate(zip(hist_x1, hist_x2)):
            if val1 != 0:
                ax.scatter(bins_x1[i], val1, marker='o', s=100, color='blue')
                ax.vlines(bins_x1[i], 0, val1, linestyle='--', color='blue', alpha=0.5)
            if val2 != 0:
                ax.scatter(bins_x2[i], val2, marker='^', s=100, color='orange')
                ax.vlines(bins_x2[i], 0, val2, linestyle='--', color='orange', alpha=0.5)
                
        ax.scatter([], [], label='Delta_1', marker='o', s=100, color='blue')
        ax.scatter([], [], label='Delta_2', marker='^', s=100, color='orange')
        ax.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Delta Values Frequency 2DPLS Cascade')
        plt.show()
            
        return delta_x1, delta_y1, delta_x2, delta_y2, W_x1, W_y1, W_x2, W_y2, U, V, time_spend
            
        
    def reconstruct(W_x1, W_x2, W_y1, W_y2, U, V, counted_mean_x, counted_mean_y, K = 2):
        #U*W
        centrated_matrix_x = np.zeros_like(U)
        centrated_matrix_y = np.zeros_like(V)
        for i in range(0,K):
            centrated_matrix_x[i] = W_x1 @ U[i] @ W_x2.T
            centrated_matrix_y[i] = W_y1 @ V[i] @ W_y2.T
        #psi+UW
        original_matrix_x = np.zeros_like(U)
        original_matrix_y = np.zeros_like(V)
        for i in range(0,K):
            original_matrix_x[i] = centrated_matrix_x[i] + counted_mean_x
            original_matrix_y[i] = centrated_matrix_y[i] + counted_mean_y
        return original_matrix_x, original_matrix_y
        
    
    def threshold(U, V, K = 4, accuracy_coef = 1):
        distances = []
        for i in range(0,K):
            for j in range(i + 1, K):
                distance = np.max(np.abs(U[i] - V[j]))
                distances.append(distance*(accuracy_coef))
        # Нахождение максимального расстояния
        max_distance = np.max(distances)
        return max_distance  
        
    #2 concatinate
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    _,x,y = X.shape
    _,k,h = Y.shape
    #3 mean
    counted_mean_x = mean(X)
    counted_mean_y = mean(Y)
    #4 centrate
    centrated_matrix_x = centrate(X, counted_mean_x, K=K)
    centrated_matrix_y = centrate(Y, counted_mean_y, K=K)
    centrated_matrix_x = np.array(centrated_matrix_x, dtype=np.float64)
    centrated_matrix_y = np.array(centrated_matrix_y, dtype=np.float64)
    #5 covariate
    #6 eigenvalues and eigenvectors
    delta_x1, delta_y1, delta_x2, delta_y2, W_x1, W_y1, W_x2, W_y2, U, V, time_spend_cascade = covariate_count_eig_weight_cascade(centrated_matrix_x, centrated_matrix_y, K)

    delta_x1, delta_y1, delta_x2, delta_y2, W_x1, W_y1, W_x2, W_y2, U, V, time_spend_parallel = covariate_count_eig_weight_parallel(centrated_matrix_x, centrated_matrix_y, K)

    #9 thereshould
    thereshould = threshold(U,V, K = K, accuracy_coef=accuracy_coef)
    
    
    print("Time spend parallel pls:")
    print(time_spend_parallel)
    print("Time spend cascade pls:")
    print(time_spend_cascade)
    print("Max distance")
    print(thereshould)
    
#    def complex_to_list(complex_num):
#        return [complex_num.real, complex_num.imag]
#    def vector_of_matrices_to_json(v):
#        json_data = []
#        for matrix in v:
#            json_data.append([[complex_to_list(x) for x in row] for row in matrix])
#        return json_data
#    def vector_of_matrices_to_json_non_complex(v):
#        json_data = []
#        for matrix in v:
#            json_data.append(matrix.tolist())
#        return json_data
#    data = {
#        "counted_mean_x": counted_mean_x.tolist(),
#        "counted_mean_y": counted_mean_y.tolist(),
#        "W_x1": [[complex_to_list(x) for x in row] for row in W_x1],
#        "W_y1": [[complex_to_list(x) for x in row] for row in W_y1],
#        "W_x2": [[complex_to_list(x) for x in row] for row in W_x2],
#        "W_y2": [[complex_to_list(x) for x in row] for row in W_y2],
#        "U": vector_of_matrices_to_json(U),
#        "V": vector_of_matrices_to_json(V),
#
#        "thereshould": thereshould
#    }
#
#    with open("data_PLS.json", "w") as json_file:
#        json.dump(data, json_file)
#
#    #8 reconstruct
    reconstructed_x, reconstructed_y = reconstruct(W_x1, W_x2, W_y1, W_y2, U, V, counted_mean_x, counted_mean_y, K = K)
#
#    data_re = {
#        "reconstructed_x": vector_of_matrices_to_json_non_complex(reconstructed_x),
#        "reconstructed_y": vector_of_matrices_to_json_non_complex(reconstructed_y),
#    }
#    
#    with open("data_PLS_re.json", "w") as json_file:
#        json.dump(data_re, json_file)
        
    print("Train is finished. All data is saved to data_pls.json and data_pls_re.json")    

    return reconstructed_x, reconstructed_y, counted_mean_x, counted_mean_y, W_x1, W_y1, W_x2, W_y2, U, V, thereshould

def check(track_matrix, counted_mean_x, counted_mean_y, W_x1, W_y1, W_x2, W_y2, U, V, thereshould, K = 4, switch = False, accuracy_coef = 1):
    if switch == False:
        counted_mean_x = np.array(counted_mean_x, dtype=np.float64)
        Fi_x = (track_matrix - counted_mean_x)
        U_weight = W_x1.T @ Fi_x @ W_x2
        closest_vector1, current_distance1 = find_closest_vector1(U_weight, U, K = K, accuracy_coef = accuracy_coef)
        print("Current distance X->Y")
        print(current_distance1)
        if current_distance1 > thereshould:
            print("PHOTO 1 IS INVALID, TRY TO INCREASE THE ACCURACY RATE")
            return None
        centrated_matrix_y = W_x1 @ closest_vector1 @ W_x2.T
        closest_matrix_y = centrated_matrix_y + counted_mean_x
        return closest_matrix_y
    if switch == True:
        counted_mean_y = np.array(counted_mean_y, dtype=np.float64)
        Fi_y = (track_matrix - counted_mean_y)
        V_weight = W_y1.T @ Fi_y @ W_y2
        closest_vector2, current_distance2 = find_closest_vector2(V_weight, V, K = K, accuracy_coef = accuracy_coef)
        print("Current distance Y->X")
        print(current_distance2)
        if current_distance2 > thereshould:
            print("PHOTO 2 IS INVALID, TRY TO INCREASE THE ACCURACY RATE")
            return None
        centrated_matrix_x = W_y1 @ closest_vector2 @ W_y2.T
        closest_matrix_x = centrated_matrix_x + counted_mean_y
        return closest_matrix_x
        

def PLS_2D(K = 2, X = None, Y = None, G = None, L = None, accuracy_coef = 2):
    #DECREASE ACCURACY_COEF -> SMALLER CHANCE TO FIND ARRAY
    #K - number of matrixes
    #X - 1 class matrix
    #Y - 2 class matrix
    #G - matix for recontruct
    #switch - if False -> from x to y (from 1 class to 2 class)
    #switch - if True -> from y to x (from 2 class to 1 class)
    if X is None:
    #1 make vectors
        X = [None] * K
        X[0] = [[-2,3, -6],[ 3,-2, 3]]
        X[1] = [[ 2, 3,-5],[ 7,-3, 2]]
    
    if Y is None:
        Y = [None] * K
        Y[0] = [[ 5, 7, 5],[-3,-2, 9]]
        Y[1] = [[ 6,-7, 8],[-3, 3,-12]]
    
    if G is None:
        G = [None]
        G = [[ 2, 3,-5],[ 7,-3, 2]]
    
    if L is None:
        L = [None]
        L = [[-5, 0, 3],[ -3,-3, 1]]
    try:
        with open("data_PLS.json", "r") as json_file:
            data = json.load(json_file)
        counted_mean_x = np.array(data["counted_mean_x"])
        counted_mean_y = np.array(data["counted_mean_y"])
        W_x1 = np.array([[complex(x[0], x[1]) for x in row] for row in data["W_x1"]], dtype=np.complex128)
        W_y1 = np.array([[complex(x[0], x[1]) for x in row] for row in data["W_y1"]], dtype=np.complex128)
        W_x2 = np.array([[complex(x[0], x[1]) for x in row] for row in data["W_x2"]], dtype=np.complex128)
        W_y2 = np.array([[complex(x[0], x[1]) for x in row] for row in data["W_y2"]], dtype=np.complex128)
        U = np.array([[[complex(x[0], x[1]) for x in row] for row in matrix] for matrix in data["U"]], dtype=np.complex128)
        V = np.array([[[complex(x[0], x[1]) for x in row] for row in matrix] for matrix in data["V"]], dtype=np.complex128)
        thereshould = data["thereshould"]
        print("Max distance")
        print(thereshould)
        
        with open("data_PLS_re.json", "r") as json_file:
            data = json.load(json_file)
        reconstructed_x = np.array([[[x for x in row] for row in matrix] for matrix in data["reconstructed_x"]], dtype=np.float64)
        reconstructed_y = np.array([[[x for x in row] for row in matrix] for matrix in data["reconstructed_y"]], dtype=np.float64)
        
    except (FileNotFoundError, json.JSONDecodeError):
        print("File data.json not found. Train started.")
        reconstructed_x, reconstructed_y, counted_mean_x, counted_mean_y, W_x1, W_y1, W_x2, W_y2, U, V, thereshould = train(X, Y, K = K, accuracy_coef=1)
        
    res1 = check(G, counted_mean_x, counted_mean_y, W_x1, W_y1, W_x2, W_y2, U, V, thereshould, K = K, switch = False, accuracy_coef = accuracy_coef)
    res1 = np.array(res1, dtype = np.float64)
    res2 = check(L, counted_mean_x, counted_mean_y, W_x1, W_y1, W_x2, W_y2, U, V, thereshould, K = K, switch = True, accuracy_coef = accuracy_coef)
    res2 = np.array(res2, dtype = np.float64)
    for i in range(0,K):
        reconstructed_x[i] = np.array(reconstructed_x[i], dtype = np.float64)
        reconstructed_y[i] = np.array(reconstructed_y[i], dtype = np.float64)
    return res1, res2,reconstructed_x, reconstructed_y

if __name__ == "__main__":
    res1, res2, reconstructed_x, reconstructed_y = PLS_2D()
    print("RES1:")
    print(res1)   
    print("RES2:")
    print(res2)   
    print("X")
    print(reconstructed_x)
    print("Y:")
    print(reconstructed_y)
    

    
    
