"""
Métodos numéricos para problemas de valor inicial  y' = f(t, y),  y(t0) = y0.

Ambas funciones trabajan tanto con ecuaciones escalares como con sistemas:
y0 puede ser un número o un arreglo de NumPy, y f debe devolver algo del
mismo tamaño que y.

Proyecto final - Ecuaciones Diferenciales 1 - Grupo 5
"""
import numpy as np


def heun(f, t0, y0, tf, h):
    """Método de Heun (Euler mejorado), orden 2.

    k1 = f(t_n, y_n)
    k2 = f(t_n + h, y_n + h*k1)
    y_{n+1} = y_n + (h/2)*(k1 + k2)
    """
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]))
        k2 = np.asarray(f(t[i] + h, y[i] + h * k1))
        y[i + 1] = y[i] + 0.5 * h * (k1 + k2)
    return t, y


def rk4(f, t0, y0, tf, h):
    """Método de Runge-Kutta clásico de cuarto orden.

    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h,   y_n + h*k3)
    y_{n+1} = y_n + (h/6)*(k1 + 2k2 + 2k3 + k4)
    """
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1,) + np.shape(y0))
    y[0] = y0
    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]))
        k2 = np.asarray(f(t[i] + h / 2, y[i] + h * k1 / 2))
        k3 = np.asarray(f(t[i] + h / 2, y[i] + h * k2 / 2))
        k4 = np.asarray(f(t[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
