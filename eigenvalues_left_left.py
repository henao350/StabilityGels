# run_eigen_left_fixed.py
from eigenvalues_gels import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

if __name__ == "__main__":
    # Parámetros por defecto (los puedes cambiar)
    gel = gelsStability(phi0=0.9,
                        gamma=1e-3,
                        chi=0.348,
                        epsilon=1.62/90.0,
                        L=1)

    # Si discriminant < 0, no tiene Upsilon reales
    if gel.discriminant < 0:
        print("P(Upsilon) tiene discriminante < 0: no hay Upsilon_0/Upsilon_1 reales.")
        sys.exit(1)

    Upsilon_0 = gel.Upsilon_0
    Upsilon_1 = gel.Upsilon_1
    b22 = gel.b22
    L = gel.L

    # Rango de Upsilon donde buscar (evitar los límites exactos)
    horizon_for_Upsilon = 1e-6
    Upsilon_min = float(Upsilon_0) + horizon_for_Upsilon
    #Upsilon_min = float(Upsilon_1) - 10*horizon_for_Upsilon
    Upsilon_max = float(Upsilon_1) - horizon_for_Upsilon
    N_points = 200

    if Upsilon_min >= Upsilon_max:
        print("Rango inválido para Upsilon (Upsilon_min >= Upsilon_max).")
        sys.exit(1)

    print(f'Upsilon_min={Upsilon_min:.6g}, Upsilon_max={Upsilon_max:.6g}')

    f = lambda Upsilon: f_left_left(gel, Upsilon)

    Upsilon_range = np.linspace(Upsilon_min, Upsilon_max, N_points)
    f_values = np.array([f(U) for U in Upsilon_range])

    # Detectar cambios de signo y refinar con brentq
    roots = []
    for i in range(len(f_values)-1):
        y1 = f_values[i]; y2 = f_values[i+1]
        if np.isnan(y1) or np.isnan(y2):
            continue
        if y1 * y2 < 0.0:
            a = Upsilon_range[i]; b = Upsilon_range[i+1]
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)
                roots.append(root)
            except Exception:
                # si falla brentq, lo ignoramos (suele ocurrir por singularidades)
                continue

    # Mostrar resultados
    print(f"Raíces encontradas en rango: {roots}")

    # Graficar
    plt.figure(figsize=(8, 5))
    # logaritmic scale for both axes in the plot
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(-Upsilon_range, f_values, label=r'$f(\Upsilon)$')
    plt.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    #for root in roots:
    #    plt.scatter(-root, 0.0, color='red', s=60, zorder=5)
    plt.xlabel(r'$\Upsilon$')
    plt.ylabel(r'$f(\Upsilon)$')
    plot_label = r'$f(\Upsilon)=f_{\text{left-left}}(\Upsilon)$' + rf', $L={L}$'
    plt.title(plot_label)
    plt.legend()
    plt.grid(True)
    plot_filename = 'eigenvalues_plot-left-left.png'
    plt.savefig(plot_filename, dpi=300)
    print("Gráfico guardado en:", plot_filename)
    # plt.show()  # descomenta si quieres verlo interactivo
