from matplotlib import pyplot as plt

def least_squares_fit(x, y, degree):
    # Create the design matrix
    X = []
    for i in range(len(x)):
        row = [x[i]**j for j in range(degree + 1)]
        X.append(row)
    
    # Transpose the design matrix
    XT = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    
    # Multiply XT and X
    XTX = [[sum(XT[i][k] * X[k][j] for k in range(len(X))) for j in range(len(X[0]))] for i in range(len(XT))]
    
    # Multiply XT and y
    XTy = [sum(XT[i][j] * y[j] for j in range(len(y))) for i in range(len(XT))]
    
    # Solve the system of linear equations XTX * a = XTy
    a = gauss_jordan(XTX, XTy)
    
    return a

def gauss_jordan(A, b):
    n = len(A)
    M = [A[i] + [b[i]] for i in range(n)]
    
    for i in range(n):
        # Make the diagonal contain all 1's
        div = M[i][i]
        for j in range(n + 1):
            M[i][j] /= div
        
        # Make the other rows contain 0's
        for k in range(n):
            if k != i:
                mult = M[k][i]
                for j in range(n + 1):
                    M[k][j] -= mult * M[i][j]
    
    return [M[i][-1] for i in range(n)]

# Example usage:
x = [0, 0.15, 0.31, 0.5, 0.6, 0.75]
y = [1.0, 1.004, 1.031, 1.117, 1.223, 1.422]

# Fit a linear polynomial (degree 1)
coefficients_linear = least_squares_fit(x, y, 1)
print("Linear fit coefficients:", coefficients_linear)

# Fit a quadratic polynomial (degree 2)
coefficients_quadratic = least_squares_fit(x, y, 2)
print("Quadratic fit coefficients:", coefficients_quadratic)

plt.figure()
plt.plot(x, y, 'ro', label='Data')
xs = [i / 100 for i in range(100)]
plt.plot(xs, [coefficients_linear[0] + coefficients_linear[1] * xi for xi in xs], label='Linear fit')
plt.plot(xs, [coefficients_quadratic[0] + coefficients_quadratic[1] * xi + coefficients_quadratic[2] * xi**2 for xi in xs], label='Quadratic fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
