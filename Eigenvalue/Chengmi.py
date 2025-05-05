#65210720
import numpy as np

def CM(A,z0):   #乘幂法
    z_old = z0
    for i in range(100):
        y = np.dot(A,z_old)
        m = max(y)
        z = y/m
        if max(np.abs(z-z_old)) < 1e-20:
            return [m,z]
        z_old = z
        
        
def CCM(A, z0):   #反乘幂法
    z_old = z0
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None    
    for i in range(100):
        y = np.dot(A_inv,z_old)
        m = max(y)
        z = y / m
        if np.linalg.norm(z - z_old)< 1e-20:
            return [1 / m, z]
        z_old = z
        
def compare_in(A,z):   #判断z是否在A中，方向相同即为True
    for z0 in A:
        z0 = z0 / np.linalg.norm(z0)
        z = z / np.linalg.norm(z)
        if np.allclose(z0, z, rtol=1e-10):
            return True
    return False

def eigenvalues(A,z0,p_min=-20,p_max=20,dp=0.5):  #反乘幂位移法
    eigenvalues = []
    eigenvectors = []
    for p in np.arange(p_min, p_max, dp):
        B = A - np.eye(A.shape[0]) * p
        result = CCM(B,z0)
        #print(p,result)
        if result == None:
            continue
        if compare_in(eigenvectors, result[1]):
            continue
        else:
            eigenvalues.append(result[0]+p)
            eigenvectors.append(result[1])
    return [eigenvalues, eigenvectors]
      


if __name__ == "__main__":
    #基于例子的测试反乘幂法加位移
    A = np.array([[2., 3., 2.],
                  [10., 3., 4.],
                  [3., 6., 1.]])
    z0 = np.array([0., 0., 1.])
    eigenvalues_result = eigenvalues(A, z0)
    print("【测试】")
    print(A)
    print("Eigenvalue",'\t',"Eigenvector")
    for i in range(len(eigenvalues_result[0])):
        print(eigenvalues_result[0][i],'\t',eigenvalues_result[1][i])
    print()
    
    
    #练习
    print("【练习】")
    A = np.array([[1.6, 2, 3],
                  [2, 3.6, 4.],
                  [3., 4, 5.6]])
    z0 = np.array([0., 0., 1.])
    eigenvalues_result = eigenvalues(A, z0, p_min=-20, p_max=20, dp=0.1)
    print(A)
    print("乘幂法: ", CM(A, z0))
    print("反乘幂法:")
    print("Eigenvalue",'\t',"Eigenvector")
    for i in range(len(eigenvalues_result[0])):
        print(eigenvalues_result[0][i],'\t',eigenvalues_result[1][i])