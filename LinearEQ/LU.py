import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    
    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def solve_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x



def plu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy()
    P = np.eye(n)

    for i in range(n):
        # Find the row with the maximum value in the current column
        max_row = np.argmax(abs(U[i:, i])) + i
        if i != max_row:
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        L[i][i] = 1
        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            U[j, i:] -= L[j][i] * U[i, i:]

    return P, L, U

def solve_plu(A, b):
    P, L, U = plu_decomposition(A)
    b = np.dot(P, b)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


A = np.array([[1, 1, 3], [5, 3, 1], [2, 3, 1]])
b = np.array([2, 3, -1])
x = solve_lu(A, b)
print(x)


A = np.array([[1., 1, 3], [5, 3, 1], [2, 3, 1]])
b = np.array([2., 3, -1])
x = solve_plu(A, b)
print(x)

A=np.array([[0.02, 61.3] , [3.43, -8.5]])
b=np.array([61.5, 25.8])
x = solve_lu(A, b)
print(x)

A=np.array([[0.02, 61.3] , [3.43, -8.5]])
b=np.array([61.5, 25.8])
x = solve_plu(A, b)
print(x)