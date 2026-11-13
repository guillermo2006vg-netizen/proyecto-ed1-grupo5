"""
Casos de prueba del proyecto: compara Heun y RK4 contra la solución analítica
(casos 1 a 3) y contra solve_ivp de SciPy (caso 4, Van der Pol).

Uso:  python casos.py
Genera las figuras en fig/ y los resultados numéricos en resultados.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from metodos import heun, rk4

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "fig")
os.makedirs(FIG, exist_ok=True)

PASOS = [0.2, 0.1, 0.05]

# ---------------------------------------------------------------- estilo Excel
AZUL, NARANJA, GRIS, AMARILLO = "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#BFBFBF",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#D9D9D9",
    "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def estadisticos(num, exacta):
    """Error máximo, error absoluto medio y raíz del error cuadrático medio."""
    e = np.abs(num - exacta)
    return {"emax": float(e.max()), "mae": float(e.mean()),
            "rmse": float(np.sqrt(np.mean(e ** 2)))}


# ======================================================================
# Caso 1:  y' = y - t^2 + 1,  y(0) = 0.5,  0 <= t <= 2
# ======================================================================
f1 = lambda t, y: y - t ** 2 + 1
y1 = lambda t: (t + 1) ** 2 - 0.5 * np.exp(t)

# ======================================================================
# Caso 2:  y'' + 4y = cos(3t),  y(0) = 0, y'(0) = 0,  0 <= t <= 10
#          con v = y':  y' = v,  v' = cos(3t) - 4y
# ======================================================================
f2 = lambda t, u: np.array([u[1], np.cos(3 * t) - 4 * u[0]])
y2 = lambda t: (np.cos(2 * t) - np.cos(3 * t)) / 5

# ======================================================================
# Caso 3:  x' = -0.5x + 2y,  y' = -2x - 0.5y,  x(0) = 1, y(0) = 0
# ======================================================================
f3 = lambda t, u: np.array([-0.5 * u[0] + 2 * u[1], -2 * u[0] - 0.5 * u[1]])
x3 = lambda t: np.exp(-t / 2) * np.cos(2 * t)
yy3 = lambda t: -np.exp(-t / 2) * np.sin(2 * t)

# ======================================================================
# Caso 4:  Van der Pol, mu = 1:  x' = y,  y' = mu(1 - x^2)y - x
# ======================================================================
MU = 1.0
f4 = lambda t, u: np.array([u[1], MU * (1 - u[0] ** 2) * u[1] - u[0]])
CI_VDP = [(2.0, 0.0), (0.5, 0.0), (0.0, 3.0), (-3.0, 2.0)]

res = {}

# ---------------------------------------------------------------- casos 1-3
casos = {
    "c1": (f1, 0.5, 2.0, lambda t: y1(t), lambda Y: Y),
    "c2": (f2, np.array([0.0, 0.0]), 10.0, lambda t: y2(t), lambda Y: Y[:, 0]),
    "c3": (f3, np.array([1.0, 0.0]), 10.0,
           lambda t: np.column_stack([x3(t), yy3(t)]), lambda Y: Y),
}
for nombre, (f, ci, tf, exacta, comp) in casos.items():
    res[nombre] = {}
    for h in PASOS:
        for met, fun in (("heun", heun), ("rk4", rk4)):
            t, Y = fun(f, 0.0, ci, tf, h)
            num, ex = comp(Y), exacta(t)
            if nombre == "c3":   # error como distancia euclidiana en el plano
                e = np.linalg.norm(num - ex, axis=1)
                est = {"emax": float(e.max()), "mae": float(e.mean()),
                       "rmse": float(np.sqrt(np.mean(e ** 2)))}
            else:
                est = estadisticos(num, ex)
            res[nombre][f"{met}_{h}"] = est

# Tabla del caso 1 con h = 0.2
t, Yh = heun(f1, 0, 0.5, 2, 0.2)
_, Yr = rk4(f1, 0, 0.5, 2, 0.2)
res["tabla_c1"] = [[float(a), float(b), float(c), float(d)]
                   for a, b, c, d in zip(t, y1(t), Yh, Yr)]

# Tabla del caso 2 con h = 0.1 en t = 1, 2, ..., 10
t, Yh = heun(f2, 0, np.array([0.0, 0.0]), 10, 0.1)
_, Yr = rk4(f2, 0, np.array([0.0, 0.0]), 10, 0.1)
idx = [int(round(k / 0.1)) for k in range(1, 11)]
res["tabla_c2"] = [[float(t[i]), float(y2(t[i])), float(Yh[i, 0]), float(Yr[i, 0])]
                   for i in idx]

# Tabla del caso 3 con h = 0.1 en t = 1, 2, ..., 5
t, Yh = heun(f3, 0, np.array([1.0, 0.0]), 10, 0.1)
_, Yr = rk4(f3, 0, np.array([1.0, 0.0]), 10, 0.1)
idx = [int(round(k / 0.1)) for k in (1, 2, 3, 4, 5, 10)]
res["tabla_c3"] = [[float(t[i]), float(x3(t[i])), float(Yh[i, 0]), float(Yr[i, 0]),
                    float(yy3(t[i])), float(Yh[i, 1]), float(Yr[i, 1])] for i in idx]

# ---------------------------------------------------------------- caso 4
res["c4"] = {}
TF4 = 20.0
for ci in CI_VDP:
    ref = solve_ivp(f4, (0, TF4), ci, method="DOP853", rtol=1e-12, atol=1e-12,
                    dense_output=True)
    clave = f"{ci[0]:g},{ci[1]:g}"
    res["c4"][clave] = {}
    for h in PASOS:
        for met, fun in (("heun", heun), ("rk4", rk4)):
            t, Y = fun(f4, 0.0, np.array(ci), TF4, h)
            e = np.linalg.norm(Y - ref.sol(t).T, axis=1)
            res["c4"][clave][f"{met}_{h}"] = {
                "emax": float(e.max()), "mae": float(e.mean()),
                "rmse": float(np.sqrt(np.mean(e ** 2))), "efin": float(e[-1])}
    # amplitud del ciclo límite (máximo de x en la última parte)
    tt = np.linspace(10, 20, 10001)
    res["c4"][clave]["amp"] = float(ref.sol(tt)[0].max())

# Comparación directa en t = 20 para la condición principal (2, 0)
ref = solve_ivp(f4, (0, TF4), (2.0, 0.0), method="DOP853", rtol=1e-12, atol=1e-12)
res["c4_ref_t20"] = [float(ref.y[0, -1]), float(ref.y[1, -1])]
fila = []
for h in PASOS:
    _, Yh = heun(f4, 0, np.array([2.0, 0.0]), TF4, h)
    _, Yr = rk4(f4, 0, np.array([2.0, 0.0]), TF4, h)
    fila.append([h, float(Yh[-1, 0]), float(Yh[-1, 1]), float(Yr[-1, 0]), float(Yr[-1, 1])])
res["c4_t20"] = fila

# Orden observado p = log2(E(h)/E(h/2))
def orden(d, met):
    e = [d[f"{met}_{h}"]["emax"] for h in PASOS]
    return [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]

res["ordenes"] = {c: {m: orden(res[c], m) for m in ("heun", "rk4")}
                  for c in ("c1", "c2", "c3")}
res["ordenes"]["c4"] = {m: orden(res["c4"]["2,0"], m) for m in ("heun", "rk4")}

with open(os.path.join(AQUI, "resultados.json"), "w") as fh:
    json.dump(res, fh, indent=1)

# ====================================================================== figuras
def guardar(nombre):
    plt.savefig(os.path.join(FIG, nombre))
    plt.close()

# Fig. caso 1
t = np.linspace(0, 2, 400)
th, Yh = heun(f1, 0, 0.5, 2, 0.2)
tr, Yr = rk4(f1, 0, 0.5, 2, 0.2)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(t, y1(t), color=GRIS, lw=2.2, label="Analítica")
ax[0].plot(th, Yh, "o", color=NARANJA, ms=4, label="Heun")
ax[0].plot(tr, Yr, "s", color=AZUL, ms=3, mfc="none", label="RK4")
ax[0].set_xlabel("t"); ax[0].set_ylabel("y(t)"); ax[0].legend(loc="upper left")
ax[0].set_title("Solución, h = 0.2", fontsize=9)
ax[1].plot(th[1:], np.abs(Yh - y1(th))[1:], "-o", color=NARANJA, ms=3, label="Heun")
ax[1].plot(tr[1:], np.abs(Yr - y1(tr))[1:], "-s", color=AZUL, ms=3, label="RK4")
ax[1].set_yscale("log"); ax[1].set_xlabel("t"); ax[1].set_ylabel("Error absoluto")
ax[1].set_title("Error absoluto, h = 0.2", fontsize=9); ax[1].legend(loc="lower right")
plt.tight_layout(); guardar("caso1.png")

# Fig. caso 2
t = np.linspace(0, 10, 2000)
th, Yh = heun(f2, 0, np.array([0.0, 0.0]), 10, 0.2)
tr, Yr = rk4(f2, 0, np.array([0.0, 0.0]), 10, 0.2)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(t, y2(t), color=GRIS, lw=2.2, label="Analítica")
ax[0].plot(th, Yh[:, 0], "--", color=NARANJA, lw=1.2, label="Heun")
ax[0].plot(tr, Yr[:, 0], ":", color=AZUL, lw=1.6, label="RK4")
ax[0].set_xlabel("t"); ax[0].set_ylabel("y(t)")
ax[0].set_title("Solución, h = 0.2", fontsize=9)
ax[0].legend(loc="lower left", ncol=3, fontsize=7.5)
ax[0].set_ylim(-0.55, 0.5)
ax[1].plot(th[1:], np.abs(Yh[:, 0] - y2(th))[1:], color=NARANJA, lw=1.2, label="Heun")
ax[1].plot(tr[1:], np.abs(Yr[:, 0] - y2(tr))[1:], color=AZUL, lw=1.2, label="RK4")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1)
ax[1].set_xlabel("t"); ax[1].set_ylabel("Error absoluto")
ax[1].set_title("Error absoluto, h = 0.2", fontsize=9); ax[1].legend(loc="lower right")
plt.tight_layout(); guardar("caso2.png")

# Fig. caso 3 (plano de fase y error)
t = np.linspace(0, 10, 2000)
th, Yh = heun(f3, 0, np.array([1.0, 0.0]), 10, 0.2)
tr, Yr = rk4(f3, 0, np.array([1.0, 0.0]), 10, 0.2)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].plot(x3(t), yy3(t), color=GRIS, lw=2.2, label="Analítica")
ax[0].plot(Yh[:, 0], Yh[:, 1], "o", color=NARANJA, ms=2.5, label="Heun")
ax[0].plot(Yr[:, 0], Yr[:, 1], "s", color=AZUL, ms=2.5, mfc="none", label="RK4")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].set_aspect("equal")
ax[0].set_title("Plano de fase, h = 0.2", fontsize=9)
ax[0].legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(1.0, 1.0))
eh = np.linalg.norm(Yh - np.column_stack([x3(th), yy3(th)]), axis=1)
er = np.linalg.norm(Yr - np.column_stack([x3(tr), yy3(tr)]), axis=1)
ax[1].plot(th[1:], eh[1:], color=NARANJA, lw=1.2, label="Heun")
ax[1].plot(tr[1:], er[1:], color=AZUL, lw=1.2, label="RK4")
ax[1].set_yscale("log"); ax[1].set_xlabel("t"); ax[1].set_ylabel("Error (norma)")
ax[1].set_title("Error, h = 0.2", fontsize=9); ax[1].legend(loc="lower left")
plt.tight_layout(); guardar("caso3.png")

# Fig. columnas: error máximo por caso y paso (estilo Excel)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, (c, tit) in zip(axs, (("c1", "1.er orden"), ("c2", "2.º orden"),
                              ("c3", "Sistema lineal"))):
    x = np.arange(len(PASOS)); w = 0.38
    eh = [res[c][f"heun_{h}"]["emax"] for h in PASOS]
    er = [res[c][f"rk4_{h}"]["emax"] for h in PASOS]
    ax.bar(x - w / 2, eh, w, color=NARANJA, label="Heun")
    ax.bar(x + w / 2, er, w, color=AZUL, label="RK4")
    ax.set_yscale("log"); ax.set_xticks(x, [f"h = {h}" for h in PASOS])
    ax.set_title(tit, fontsize=9); ax.tick_params(axis="x", length=0)
axs[0].set_ylabel("Error máximo")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.06))
plt.tight_layout(); guardar("columnas_error.png")

# Fig. Van der Pol: plano de fase con las 4 CI (RK4, h = 0.05) y referencia
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
estilos = ["-", "--", "-.", ":"]
colores = [AZUL, NARANJA, GRIS, AMARILLO]
for ci, ls, col in zip(CI_VDP, estilos, colores):
    _, Y = rk4(f4, 0, np.array(ci), 20, 0.05)
    ax[0].plot(Y[:, 0], Y[:, 1], ls, color=col, lw=1.2, label=f"({ci[0]:g}, {ci[1]:g})")
    ax[0].plot(*ci, "o", color=col, ms=4)
ax[0].set_xlabel("x"); ax[0].set_ylabel("y = x'")
ax[0].set_title("Plano de fase (RK4, h = 0.05)", fontsize=9)
ax[0].legend(fontsize=7, loc="lower right", title="C. I.", title_fontsize=7, ncol=1)
ax[0].set_xlim(-3.3, 4.6)
ref = solve_ivp(f4, (0, 20), (2.0, 0.0), method="DOP853", rtol=1e-12, atol=1e-12,
                dense_output=True)
tt = np.linspace(0, 20, 2000)
ax[1].plot(tt, ref.sol(tt)[0], color=GRIS, lw=2.2, label="solve_ivp")
th, Yh = heun(f4, 0, np.array([2.0, 0.0]), 20, 0.2)
tr, Yr = rk4(f4, 0, np.array([2.0, 0.0]), 20, 0.2)
ax[1].plot(th, Yh[:, 0], "--", color=NARANJA, lw=1.1, label="Heun")
ax[1].plot(tr, Yr[:, 0], ":", color=AZUL, lw=1.6, label="RK4")
ax[1].set_xlabel("t"); ax[1].set_ylabel("x(t)")
ax[1].set_title("x(t) con C. I. (2, 0), h = 0.2", fontsize=9)
ax[1].legend(fontsize=7, loc="lower left", ncol=3); ax[1].set_ylim(-3.2, 2.6)
plt.tight_layout(); guardar("vdp_fase.png")

# Fig. convergencia de Van der Pol (columnas)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
x = np.arange(len(PASOS)); w = 0.38
eh = [res["c4"]["2,0"][f"heun_{h}"]["emax"] for h in PASOS]
er = [res["c4"]["2,0"][f"rk4_{h}"]["emax"] for h in PASOS]
ax.bar(x - w / 2, eh, w, color=NARANJA, label="Heun")
ax.bar(x + w / 2, er, w, color=AZUL, label="RK4")
ax.set_yscale("log"); ax.set_xticks(x, [f"h = {h}" for h in PASOS])
ax.set_ylabel("Diferencia máx. con solve_ivp"); ax.tick_params(axis="x", length=0)
ax.legend(fontsize=7.5, loc="upper right")
plt.tight_layout(); guardar("vdp_convergencia.png")

print(json.dumps(res["ordenes"], indent=1))
