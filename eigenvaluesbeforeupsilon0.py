import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math

# ===========================
# PARÁMETROS DEL MODELO
# ===========================
lam = 1.96
epsilon = 1.62 / 90
phi0 = 0.9
gamma = 1e-3
chi = 0.348
L = 1  # Longitud física, como pediste
horizon_for_Upsilon = 1e-7
# ===========================
# FUNCIONES AUXILIARES
# ===========================
def phi(J):
    return phi0 / J
def H(J):
    return (J - phi0) * np.log(1 - phi(J)) + phi0 * chi * (1 - phi(J))
def H_prime(J):
    return phi(J) + np.log(1 - phi(J)) + chi * phi(J)**2
def H_double_prime(J):
    return (phi(J))**2 / J * (1 / (1 - phi(J)) - 2 * chi)
# ===========================
# Coeficientes a_ij y b_ij
# ===========================
a11 = 4 * math.pi**2 * (gamma + lam**2 * H_double_prime(lam))
a22 = gamma + H_double_prime(lam)
b11 = epsilon**(-2) * gamma
b22 = 4 * math.pi**2 * gamma * epsilon**2
a12 = -math.pi * (lam * H_double_prime(lam) + H_prime(lam))
b12 = -math.pi * H_prime(lam)

# ===========================
# FUNCIÓN MAESTRA
# ===========================
def f(Upsilon):
    # Valores intermedios
    b_Upsilon = -b11*b22 + b11*Upsilon - a11*a22 + a22*Upsilon + 4*(b12-a12)**2
    a_Upsilon = a22*b11

    Delta_Upsilon = (b11**2 * b22**2 + a11**2 * a22**2 + 16 * (b12 - a12)**4 +
                     (a22 + b11)**2 * Upsilon**2 + 2 * a11 * a22 * b11 * b22 -
                     8 * a11 * a22 * (b12 - a12)**2 +
                     8 * (a22 + b11) * (b12 - a12)**2 * Upsilon -
                     2 * b11 * b22 * (a22 + b11) * Upsilon -
                     8 * b11 * b22 * (b12 - a12)**2 -
                     2 * a11 * a22 * (b11 + a22) * Upsilon -
                     4 * a22 * b11 * (a11 * b22 - a11 * Upsilon - b22 * Upsilon + Upsilon**2))

    # Argumentos de mu1 y mu3
    mu_1_arg = (b_Upsilon + np.sqrt(Delta_Upsilon)) / (2 * a_Upsilon)
    mu_3_arg = (-b_Upsilon + np.sqrt(Delta_Upsilon)) / (2 * a_Upsilon)

    if mu_1_arg <= 0 or mu_3_arg <= 0:
        return np.nan

    mu_1 = np.sqrt(mu_1_arg)
    mu_3 = np.sqrt(mu_3_arg)

    # p0 y s0
    if np.isclose(b11*mu_3**2 - a11 - Upsilon, 0) or np.isclose(b12 - a12, 0):
        return np.nan

    p0 = (-2 * (b12 - a12)) / (b11 * mu_3**2 - a11 + Upsilon)
    s0 = (b11 * mu_1**2 - a11 + Upsilon) / (-2 * (b12 - a12))

    if np.isclose(s0, 0):
        return np.nan

    # θ = μ₁
    theta = mu_1

    # ===============================
    # ECUACIÓN MAESTRA
    # ===============================
    # term1 = -mu_3 * (
    #     (p0 / s0) * (-b11 * theta**2 + 2 * b12 * s0) * (a22 * s0 + 2 * a12) +
    #     (a22 + 2 * a12 * p0) * (b11 * p0 * mu_3**2 + 2 * b12)
    # )

    # term2 = -(
    #     (mu_3**2 * p0 / theta) * (a22 + 2 * a12 * p0) * (-b11 * theta**2 + 2 * b12 * s0)
    #     - (theta / s0) * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    # ) * np.sin(theta * L) * np.sinh(mu_3 * L)

    # term3 = (
    #     (mu_3 / s0) * (a22 + 2 * a12 * p0) * (-b11 * theta**2 + 2 * b12 * s0)
    #     + mu_3 * p0 * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    # ) * np.cos(theta * L) * np.cosh(mu_3 * L)

    term1 = - (
        p0  * (-b11 * theta**2 + 2 * b12 * s0) * (a22 * s0 + 2 * a12) +
        s0*(a22 + 2 * a12 * p0) * (b11 * p0 * mu_3**2 + 2 * b12)
    )

    term2 = -(
        (mu_3**2 * p0 / theta) *s0* (a22 + 2 * a12 * p0) * (-b11 * theta**2 + 2 * b12 * s0)
        - theta  * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    ) * np.sin(theta * L) * (np.sinh(mu_3 * L)/mu_3)

    term3 = (
         (a22 + 2 * a12 * p0) * (-b11 * theta**2 + 2 * b12 * s0)
        + s0* p0 * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    ) * np.cos(theta * L) * np.cosh(mu_3 * L)

    result = term1 + term2 + term3

    return np.real(result) if np.isfinite(result) else np.nan

# ===========================
# BÚSQUEDA DE RAÍCES
# ===========================
#Upsilon_min = gamma + 1e-4
#Upsilon_max = gamma + 0.1
Upsilon_min = 4 * math.pi**2 * gamma * epsilon**2
# Upsilon_min = 4 * math.pi**2 * gamma * epsilon**2 + 0.5*horizon_for_Upsilon
# Upsilon_min = 0.0000001
Upsilon_max = Upsilon_min + horizon_for_Upsilon
# Upsilon_max = 4 * math.pi**2 * (gamma + lam**2 * H_double_prime(lam))
Upsilon_range = np.linspace(Upsilon_min, Upsilon_max, 1000)
f_values = np.array([f(U) for U in Upsilon_range])

print("Upsilon_min:", Upsilon_min)

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
plt.title(rf'Ecuación Maestra con $L={L}$: $f(\Upsilon)=0$')
plt.legend()
plt.grid(True)
# plt.show()
# save plot as .png
plt.savefig('eigenvalues_plot.png', dpi=300)

# ===========================
# IMPRIMIR RAÍCES
# ===========================
print("Raíces encontradas para f(Upsilon) = 0 (con L=1):")
for r in roots:
    print(f"Upsilon ≈ {r:.10f}")

# # ============================================================================================================
# # ============================================================================================================
# # 9 oct 2025
# # Attempt to find negative eigenvalues, in the range Upsilon < Upsilon_0
# # ============================================================================================================
# # ============================================================================================================

# # ===========================
# # Special values Upsilon_0 and Upsilon_1
# # ===========================
# # 1. cuadratic term (C2) - Coefficient of Upsilon^2
# C2 = (a22 - b11)**2
# # 2. Linear Coefficient (C1) - Coefficient of Upsilon
# term1_C1 = 8 * (a22 + b11) * (b12 - a12)**2
# term2_C1 = -2 * b11 * b22 * (a22 + b11)
# term3_C1 = -2 * a11 * a22 * (b11 + a22)
# term4_C1 = -4 * a22 * b11*(-a11-b22)
# C1 = term1_C1 + term2_C1 + term3_C1 + term4_C1
# # 3. Free Term (C0) - Constant term
# C0 = (b11**2 * b22**2 + a11**2 * a22**2 + 16 * (b12 - a12)**4 
#         - 2 * a11 * a22 * b11 * b22 
#         - 8 * a11 * a22 * (b12 - a12)**2
#         - 8 * b11 * b22 * (b12 - a12)**2)
# # --- Calcular las raíces de P(Upsilon) ---
# discriminant = C1**2 - 4*C0*C2
# print(f"Discriminant of P(Upsilon):{discriminant:.7f}")
# if discriminant < 0:
#     print("No hay raíces reales, solo complejas.")
#     root1 = (-C1 + complex(0, math.sqrt(-discriminant))) / (2*C2)
#     root2 = (-C1 - complex(0, math.sqrt(-discriminant))) / (2*C2)
# else:
#     root1 = (-C1 + math.sqrt(discriminant)) / (2*C2)
#     root2 = (-C1 - math.sqrt(discriminant)) / (2*C2)
#     Upsilon_0 = root2
#     Upsilon_1 = root1
# print(f"Raíces de P(Upsilon): {root1}, {root2}")

# ===========================
# FUNCIÓN MAESTRA
# ===========================
def f_left_left_left(Upsilon):          # left_left_left stands for Upsilon < Upsilon_0
    # Valores intermedios
    b_Upsilon = -b11*b22 + b11*Upsilon - a11*a22 + a22*Upsilon + 4*(b12-a12)**2
    a_Upsilon = a22*b11
    Delta_Upsilon = (b11**2 * b22**2 + a11**2 * a22**2 + 16 * (b12 - a12)**4 +
                     (a22 + b11)**2 * Upsilon**2 + 2 * a11 * a22 * b11 * b22 -
                     8 * a11 * a22 * (b12 - a12)**2 +
                     8 * (a22 + b11) * (b12 - a12)**2 * Upsilon -
                     2 * b11 * b22 * (a22 + b11) * Upsilon -
                     8 * b11 * b22 * (b12 - a12)**2 -
                     2 * a11 * a22 * (b11 + a22) * Upsilon -
                     4 * a22 * b11 * (a11 * b22 - a11 * Upsilon - b22 * Upsilon + Upsilon**2))
    # Argumentos de mu1 y mu3
    mu_1_arg = (-b_Upsilon - np.sqrt(Delta_Upsilon)) / (2 * a_Upsilon)
    mu_3_arg = (-b_Upsilon + np.sqrt(Delta_Upsilon)) / (2 * a_Upsilon)
    if mu_1_arg <= 0 or mu_3_arg <= 0:
        return np.nan
    mu_1 = np.sqrt(mu_1_arg)
    mu_3 = np.sqrt(mu_3_arg)
    # p0 y s0
    if np.isclose(b11*mu_3**2 - a11 - Upsilon, 0) or np.isclose(b12 - a12, 0):
        return np.nan
    p0 = (-2 * (b12 - a12)) / (b11 * mu_3**2 - a11 + Upsilon)
    s0 = (b11 * mu_1**2 - a11 + Upsilon) / (-2 * (b12 - a12))
    if np.isclose(s0, 0):
        return np.nan
    term1 = - (
        p0  * (b11 * mu_1**2 + 2 * b12 * s0) * (a22 * s0 + 2 * a12) +
        s0*(a22 + 2 * a12 * p0) * (b11 * p0 * mu_3**2 + 2 * b12)
    )
    term2 = -(
        (mu_3**2 * p0 / mu_1) *s0* (a22 + 2 * a12 * p0) * (b11 * mu_1**2 + 2 * b12 * s0)
        + mu_1 * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    ) * np.sinh(mu_1 * L) * (np.sinh(mu_3 * L)/mu_3)
    term3 = (
         (a22 + 2 * a12 * p0) * (b11 * mu_1**2 + 2 * b12 * s0)
        + s0* p0 * (b11 * p0 * mu_3**2 + 2 * b12) * (a22 * s0 + 2 * a12)
    ) * np.cosh(mu_1 * L) * np.cosh(mu_3 * L)
    result = term1 + term2 + term3
    return np.real(result) if np.isfinite(result) else np.nan

# ===========================
# BÚSQUEDA DE RAÍCES
# ===========================
#Upsilon_min = gamma + 1e-4
#Upsilon_max = gamma + 0.1
# Upsilon_min = 4 * math.pi**2 * gamma * epsilon**2
# Upsilon_min = 0.0000001
# Upsilon_max = 4 * math.pi**2 * (gamma + lam**2 * H_double_prime(lam))
# Upsilon_max = Upsilon_0; Upsilon_min = Upsilon_max - horizon_for_Upsilon
#Upsilon_min = Upsilon_1 + horizon_for_Upsilon; Upsilon_max = Upsilon_min + 50*horizon_for_Upsilon
Upsilon_min = 0 - horizon_for_Upsilon; Upsilon_max = 0
Upsilon_range = np.linspace(Upsilon_min, Upsilon_max, 1000)
f_values = np.array([f_left_left_left(U) for U in Upsilon_range])

# print("Upsilon_min:", Upsilon_min)

# Detectar cambios de signo
roots = []
for i in range(len(f_values)-1):
    if np.isnan(f_values[i]) or np.isnan(f_values[i+1]):
        continue
    if f_values[i] * f_values[i+1] < 0:
        try:
            root = brentq(f_left_left_left, Upsilon_range[i], Upsilon_range[i+1])
            roots.append(root)
        except ValueError:
            pass

# ===========================
# GRAFICAR RESULTADO
# ===========================
plt.figure(figsize=(8, 5))
plt.plot(Upsilon_range, f_values, label=r'$f_left_left_left(\Upsilon)$', color='b')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
for root in roots:
    plt.scatter(root, 0, color='red', s=80, zorder=3)
plt.xlabel(r'$\Upsilon$')
plt.ylabel(r'$f_left_left_left(\Upsilon)$')
plt.title(rf'Ecuación Maestra con $L={L}$: $f_left_left_left(\Upsilon)=0$')
plt.legend()
plt.grid(True)
# plt.show()
# save plot as .png
plt.savefig('eigenvalues_plot-left_left_left.png', dpi=300)
