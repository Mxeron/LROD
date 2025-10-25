import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


# Latent Representation-Based Outlier Detection Using Fuzzy Rough Sets (LROD)
def LROD(X, alpha):
    X = X.T.copy()
    p, n = X.shape
    
    Z = np.zeros((n, n))
    L = np.zeros((p, p))
    E = np.zeros((p, n))
    Y = np.zeros((p, n))
    mu = 1e-6
    mu_max = 1e6
    rho = 1.1
    eps = 1e-8
    k = 0

    lam = 0.0001
    tolerance = 1e-6
    patience = 10
    counter = 0
    p_residual = 0

    def soft_thresholding(X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    def update_Z(X, L, E, Y, mu):
        XT = X.T
        A = np.dot(XT, X)
        B = np.dot(XT, (X - np.dot(L, X) - E + Y / mu))
        Z = np.linalg.inv(np.eye(A.shape[0]) + mu * A).dot(mu * B)
        return Z

    def update_L(X, Z, E, Y, mu):
        XT = X.T
        A = np.dot(X, XT)
        B = np.dot((X - np.dot(X, Z) - E + Y / mu), XT)
        L = mu * B.dot(np.linalg.inv(np.eye(A.shape[0]) + mu * A))
        return L

    def update_E(X, Z, L, Y, lam, mu):
        A = X - np.dot(X, Z) - np.dot(L, X) + Y / mu
        E = soft_thresholding(A, lam / mu)
        return E

    def update_Y(Y, X, Z, L, E, mu):
        Y_new = Y + mu * (X - np.dot(X, Z) - np.dot(L, X) - E)
        return Y_new

    max_iter = 1000
    while k < max_iter:
        Z = update_Z(X, L, E, Y, mu)
        L = update_L(X, Z, E, Y, mu)
        E = update_E(X, Z, L, Y, lam, mu)
        Y = update_Y(Y, X, Z, L, E, mu)
        mu = min(rho * mu, mu_max)
        print("k: ", k)
        residual = np.linalg.norm(X - np.dot(X, Z) - np.dot(L, X) - E, ord=np.inf)
        print("residual: ", residual)
        if abs(residual - p_residual) < tolerance:
            counter += 1
            if counter >= patience:
                break
        else:
            counter = 0
        p_residual = residual
        if residual < eps:
            break
        k += 1
    
    X_new = np.dot(X, Z) + np.dot(L, X)
    X_new = X_new.T
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)
    
    card = np.zeros((p, n))
    for k in range(p):
        fs = 1 - cdist(X_new[:,[k]], X_new[:,[k]])
        card[k] = np.sum(fs, axis=1) / n
    
    F_OS = 1 - np.sum(card, axis=0) / p
    E = X - np.dot(X, Z) - np.dot(L, X)
    E_final = E.T
    E_OS = np.sum(E_final ** 2, axis=1) ** 0.5
    scaler = MinMaxScaler()
    E_OS = scaler.fit_transform(E_OS.reshape(-1, 1))
    E_OS = E_OS[:,0]
    
    OS = alpha * F_OS + (1 - alpha) * E_OS
    
    return OS