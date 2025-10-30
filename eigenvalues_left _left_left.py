# This file is for the range:
#    Upsilon_1 < Upsilon < b_{22}
#
from eigenvalues_gels import *
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ===========================
    # PARÁMETROS DEL MODELO
    # ===========================
    # Values by defect, 
                #  phi0=0.2,
                #  gamma=1e-3,
                #  chi=0.348,
                #  epsilon=1.62/90.0,
                #  L=1
    # #####
    # First part of the parameters to be modified
    # #####
    gel = gelsStability(phi0=0.9,
                        gamma=1e-3,
                        chi=0.4,
                        epsilon=1.62/90.0,
                        L=250);
    #
    if gel.discriminant<0:
        sys.exit(1)    
    # From now on, it will be assumed that P(Upsilon) (which is defined as b^2-4ac with a, b, c from the biquadratic equation for \mu), has two real roots Upsilon0<Upsilon1 (discriminant (pf P(Upsilon))>=0)
    # In particular, gel.Upsilon0 and gel.Upsilon1 are well defined
    Upsilon_0 = gel.Upsilon_0; Upsilon_1 = gel.Upsilon_1; b22=gel.b22; L=gel.L;
    # #####
    # Second part of the parameters to be modified
    # #####
    f = lambda Upsilon : f_left(gel, Upsilon)
    plot_label = r'$f(\Upsilon)=f_{\text{left}}(\Upsilon)$, $\Upsilon<\Upsilon_0$' + rf', $L={L}$'
    plot_filename = 'eigenvalues_plot-left_left_left.png'
    horizon_for_Upsilon = 1e-7
    Upsilon_min = Upsilon_0  - 0.01*abs(Upsilon_0)
    Upsilon_max = Upsilon_0 - horizon_for_Upsilon
    N_points = 100
    #

    # ===========================
    # BÚSQUEDA DE RAÍCES
    # ===========================
    print(f'Upsilon_min={Upsilon_min:.4e}, Upsilon_max={Upsilon_max:.4e}')
    Upsilon_range = np.linspace(Upsilon_min, Upsilon_max, N_points)    
    f_values = np.array([f(U) for U in Upsilon_range])

    # Detectar cambios de signo
    roots = []
    for i in range(len(f_values)-1):
        if np.isnan(f_values[i]) or np.isnan(f_values[i+1]):
            continue
        if f_values[i] * f_values[i+1] < 0:
            try:
                root = brentq(f, Upsilon_range[i], Upsilon_range[i+1])
                roots.append(root)
            except ValueError:
                pass

    # ===========================
    # GRAFICAR RESULTADO
    # ===========================
    plt.figure(figsize=(8, 5))
    plt.plot(Upsilon_range, f_values, label=r'$f(\Upsilon)$', color='b')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for root in roots:
        plt.scatter(root, 0, color='red', s=80, zorder=3)
    plt.xlabel(r'$\Upsilon$')
    plt.ylabel(r'$f(\Upsilon)$')
    plt.title(plot_label)
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save plot as .png
    plt.savefig(plot_filename, dpi=300)

