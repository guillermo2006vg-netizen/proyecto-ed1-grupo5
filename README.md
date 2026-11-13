# Proyecto final – Ecuaciones Diferenciales 1 (Grupo 5)

Universidad del Valle de Guatemala · Segundo ciclo 2026
Alternativa 2: Métodos numéricos para resolver EDs

Integrantes:
- César Enrique Guillermo Imeri
- Guillermo Antonio Villeda Guerra
- Juan Diego González Galiano

## Contenido

| Archivo | Descripción |
|---|---|
| `metodos.py` | Funciones `heun` y `rk4` para ecuaciones escalares y sistemas |
| `casos.py` | Resuelve los cuatro casos, calcula los errores y genera las figuras |
| `requirements.txt` | Librerías necesarias |

## Casos resueltos

1. Primer orden: y' = y - t² + 1, y(0) = 0.5, 0 ≤ t ≤ 2
2. Segundo orden: y'' + 4y = cos(3t), y(0) = 0, y'(0) = 0, 0 ≤ t ≤ 10
3. Sistema lineal: x' = -0.5x + 2y, y' = -2x - 0.5y, x(0) = 1, y(0) = 0, 0 ≤ t ≤ 10
4. Van der Pol (μ = 1) con condiciones iniciales (2, 0), (0.5, 0), (0, 3) y (−3, 2), 0 ≤ t ≤ 20, comparado con scipy.integrate.solve_ivp

Cada caso se resuelve con h = 0.2, 0.1 y 0.05.

## Cómo ejecutarlo

    pip install -r requirements.txt
    python casos.py

El programa genera las figuras del informe en la carpeta fig/ y guarda los resultados numéricos en resultados.json.

## Uso de los métodos con otra ecuación

    import numpy as np
    from metodos import heun, rk4

    f = lambda t, y: np.array([y[1], -4 * y[0]])   # y'' + 4y = 0 como sistema
    t, Y = rk4(f, 0.0, np.array([1.0, 0.0]), 10.0, 0.05)
