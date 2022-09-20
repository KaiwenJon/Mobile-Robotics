import numpy as np
import matplotlib.pyplot as plt
import math

# Note: all functions must return NUMPY vectors or matrices
def question1(A, B):
    '''
    Use a foor loop to multiply these matrices
    Arguments:
        A: a matrix with shape (n,m)
        B: a matrix with shape (m,l)
    Returns:
        C: a matrix with shape (n,l)

    '''
    # your code here
    n = A.shape[0]
    m = A.shape[1]
    l = B.shape[1]
    # print(A)
    # print(B)
    # print(n, m, l)
    C = np.zeros((n, l))
    for i in range(n):
        for j in range(l):
            for k in range(m):
                C[i][j] += A[i][k]*B[k][j]
    # print(C)
    return C

def question2(A, B):
    '''
    Use a foor loop to transpose and add both matrices (transpose both matrices first and then add them)
    Arguments:
        A: a matrix with shape (n,n)
        B: a matrix with shape (n,n)
    Returns:
        C: a matrix with shape (n,n)

    '''
    # your code here
    # print(A)
    # print(B)
    n, m = A.shape[0:2]
    # At = np.zeros((m, n))
    # Bt = np.zeros((m, n))
    Ct = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            Ct[i][j] = A[j][i] + B[j][i]
    # print(Ct)
    return Ct


def question3(A, b):
    '''
    Solve the equation Ax = b
    Arguments:
        A: a matrix with shape (n,n)
        b: a vector with shape (n,1)
    Returns:
        x: a vector with shape (n,1) if the equation has one solution
        0: if the equation has no solution or has infinite solutions

    '''
    # z = np.copy(b)
    # your code here
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    # print(A)
    # print(b)
    # try:
    #     x = np.linalg.solve(A, b)
    #     return x
    # except Exception as e:
    #     # print(e)
    #     return 0
    n = A.shape[0]
    for i in range(n):
        

        # print("==============")
        # print(A)
        # print(b)
        # print("---------------")
        # we want to make A[n][n] 1, and A[n][others] 0.
        # make A[n][n] to 1
        
        coeff = A[i][i]
        if(coeff == 0):
            # swap row with the next row
            found = False
            for below in range(i, n):
                if(A[below][i] != 0):
                    # swap below and i
                    found = True
                    for s in range(n):
                        tmp = A[i][s]
                        A[i][s] = A[below][s]
                        A[below][s] = tmp
                    tmp = b[i][0]
                    b[i][0] = b[below][0]
                    b[below][0] = tmp
                    break
            if(not found):
                continue
        coeff = A[i][i]
        for j in range(n):
            A[i][j] /= coeff
        b[i][0] /= coeff

        # make A[n][others] 0.
        for j in range(n):
            # for every other row
            if(j == i):
                continue
            coeff = A[j][i]
            # print(coeff)
            for k in range(n):
                # for all element on row j
                A[j][k] -= coeff*A[i][k]
            b[j][0] -= coeff*b[i][0]
    # print(A)
    # print(b)

    for i in range(n):
        all_zero = True
        for j in range(n):
            if A[i][j] != 0:
                all_zero = False
                break
        if(all_zero):
            return 0
    # print(b)
    return b

def question4(A):
    '''
    Compute the eigenvalues of A
    Arguments:
        A: a matrix with shape (n,n)
    Returns:
        C: the inverse of A if the real part of all eigenvalues of A are negative. Shape (n,n)
        0: otherwise

    '''
    w, v = np.linalg.eig(A)
    for eig in w:
        if(eig.real >= 0):
            return 0
            pass
    n = A.shape[0]
    A = np.concatenate((A, np.identity(n)), axis=1)
    # print(A)
    for i in range(n):
        # print("==============")
        # print(A)
        # print(b)
        # print("---------------")
        # we want to make A[n][n] 1, and A[n][others] 0.
        # make A[n][n] to 1
        
        coeff = A[i][i]
        if(coeff == 0):
            # swap row with the next row
            found = False
            for below in range(i, n):
                if(A[below][i] != 0):
                    # swap below and i
                    found = True
                    for s in range(2*n):
                        tmp = A[i][s]
                        A[i][s] = A[below][s]
                        A[below][s] = tmp
                    break
            if(not found):
                continue
        coeff = A[i][i]
        for j in range(2*n):
            A[i][j] /= coeff

        # make A[n][others] 0.
        for j in range(n):
            # for every other row
            if(j == i):
                continue
            coeff = A[j][i]
            # print(coeff)
            for k in range(2*n):
                # for all element on row j
                A[j][k] -= coeff*A[i][k]
    A_inv = A[:, n:]
    # print(A_inv)
    return A_inv
    # print(A)

def question5(N = 10, deltaT = 0.01):
    '''
    Integrate the following function from 0 to N seconds using Euler integration with x(0) = 1
    and time step of deltaT seconds
    x' = -2x^3 + sin(0.5t)x
    Arguments:
        N: Number of seconds to integrate
        deltaT: time step for integration
        
    Returns:
        x: a vector with shape (N/deltaT,1)

    '''
    # your code here
    x = np.zeros((int(N/deltaT), 1))
    x[0] = 1
    n = 1
    for i in np.arange(0, N, deltaT, dtype=float):
        if(n >= len(x)):
            break
        x[n] = x[n-1] + deltaT*(-2*pow(x[n-1], 3) + math.sin(0.5*n*deltaT)*x[n-1])
        n += 1
    t = np.arange(0, N, deltaT, dtype=float)
    # plt.title("Euler Method Integration")
    # plt.xlabel("Time (second)")
    # plt.ylabel("x position")
    # plt.plot(t,x) 
    # plt.show()
    return x

if __name__ == '__main__':
    # you can use the main function to test your functions. Not graded.
    # A = np.array([[0, 1, 2],
    #               [3, 4, 5]])
    # B = np.array([[0, 1, 0, 5],
    #               [2, 3, 0, 1],
    #               [4, 5, 0, 2]])
    # A = np.array([[0, 1, 2],
    #               [3, 4, 5]])
    # B = np.array([[0, 1, 0],
    #               [4, 5, 0]])
    # A = np.array([[2.4, 1, -2],
    #               [0, 1, 0],
    #               [0, 1, 0]], dtype=float)
    # b = np.array([[2],
    #               [2],
    #               [2]], dtype=float)
    A = np.array([[2, 1],
                  [-5, 6]], dtype=float)
    b = np.array([[1],
                  [2]], dtype=float)
    x = np.linalg.solve(A, b)
    print(x)
    # A = np.array([[2, 1, -2],
    #               [1, 0, 0],
    #               [0, 1, 0]], dtype=float)
    # x = np.linalg.inv(A)

    output = question3(A, b)
    
    print(output)
    # question5()
