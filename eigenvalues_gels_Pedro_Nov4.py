# eigenvalues_gels_Pedro_Nov4_fixed.py
import numpy as np
import math
from scipy.optimize import brentq

class gelsStability:
    """
    Clase que contiene los parámetros del gel y calcula valores auxiliares
    (lambda de equilibrio, coeficientes, raíces de P(Upsilon), etc.)
    """
    def __init__(self,
                 phi0=0.2,
                 gamma=1e-3,
                 chi=0.348,
                 epsilon=1.62/90.0,
                 L=1.0):
        # parámetros
        self.phi0 = float(phi0)
        self.gamma = float(gamma)
        self.chi = float(chi)
        self.epsilon = float(epsilon)
        self.L = float(L)

        # Helpers como métodos
        H_prime = self.H_prime
        H_double_prime = self.H_double_prime

        # Calcular lambda uniaxial homogénea resolviendo gamma*lambda + H'(lambda) = 0
        f_aux_lambda_uniaxial = lambda _lambda: self.gamma * _lambda + H_prime(_lambda)
        # Intentamos búsqueda en rango razonable
        try:
            lam = brentq(f_aux_lambda_uniaxial, 1.0, 8.0)
        except ValueError as e:
            # si brentq falla, re-lanza con mensaje claro
            raise RuntimeError("No se pudo encontrar lambda uniaxial en [1,8]. "
                               "Revisa parámetros iniciales. Mensaje: " + str(e))
        self.lam = float(lam)

        # Coeficientes a_ij y b_ij (valores numéricos)
        self.a11 = 4.0 * math.pi**2 * (self.gamma + (self.lam**2) * H_double_prime(self.lam))
        self.a22 = self.gamma + H_double_prime(self.lam)
        self.b11 = (self.epsilon**(-2)) * self.gamma
        self.b22 = 4.0 * math.pi**2 * self.gamma * (self.epsilon**2)
        self.a12 = -math.pi * (self.lam * H_double_prime(self.lam) + H_prime(self.lam))
        self.b12 = -math.pi * H_prime(self.lam)

        # C2, C1, C0 para P(Upsilon)
        a11 = self.a11; a22 = self.a22; b11 = self.b11; b22 = self.b22; a12 = self.a12; b12 = self.b12

        C2 = (a22 - b11)**2

        term1_C1 = 8.0 * (a22 + b11) * (b12 - a12)**2
        term2_C1 = -2.0 * b11 * b22 * (a22 + b11)
        term3_C1 = -2.0 * a11 * a22 * (b11 + a22)
        term4_C1 = -4.0 * a22 * b11 * (-a11 - b22)
        C1 = term1_C1 + term2_C1 + term3_C1 + term4_C1

        C0 = (b11**2 * b22**2 + a11**2 * a22**2 + 16.0 * (b12 - a12)**4
              - 2.0 * a11 * a22 * b11 * b22
              - 8.0 * a11 * a22 * (b12 - a12)**2
              - 8.0 * b11 * b22 * (b12 - a12)**2)

        discriminant = C1**2 - 4.0 * C0 * C2
        self.discriminant = discriminant
        self.C0 = C0; self.C1 = C1; self.C2 = C2

        if discriminant < 0:
            root1 = (-C1 + complex(0, math.sqrt(-discriminant))) / (2.0 * C2)
            root2 = (-C1 - complex(0, math.sqrt(-discriminant))) / (2.0 * C2)
            # no definimos Upsilon_0/1 reales
            self.Upsilon_0 = None
            self.Upsilon_1 = None
        else:
            root1 = (-C1 + math.sqrt(discriminant)) / (2.0 * C2)
            root2 = (-C1 - math.sqrt(discriminant)) / (2.0 * C2)
            # ordenar para que Upsilon_0 < Upsilon_1
            self.Upsilon_0 = min(root1, root2)
            self.Upsilon_1 = max(root1, root2)

        self.root1 = root1
        self.root2 = root2

        # Impresión resumida
        self.print_values()

    def print_values(self):
        print(f'lambda = {self.lam:.6g}, epsilon={self.epsilon:.6g}, phi0={self.phi0}, gamma={self.gamma}, chi={self.chi}, L={self.L}')
        print(f'a11={self.a11:.6g}, a22={self.a22:.6g}, b11={self.b11:.6g}, b22={self.b22:.6g}')
        print(f'a12={self.a12:.6g}, b12={self.b12:.6g}')
        if self.discriminant < 0:
            print("Discriminant < 0: P(Upsilon) tiene raíces complejas.")
        else:
            print(f"Raíces reales de P(Upsilon): Upsilon_0={self.Upsilon_0:.6g}, Upsilon_1={self.Upsilon_1:.6g}")

    # ===========================
    # FUNCIONES AUXILIARES
    # ===========================
    def phi(self, J: float) -> float:
        """phi(J) = phi0 / J"""
        return self.phi0 / J

    def H(self, J: float) -> float:
        """Energía H(J) (numérico)"""
        phiJ = self.phi(J)
        # proteger log domain
        if phiJ >= 1.0:
            return np.nan
        return (J - self.phi0) * np.log(1.0 - phiJ) + self.phi0 * self.chi * (1.0 - phiJ)

    def H_prime(self, J: float) -> float:
        """Derivada H'(J) numérica"""
        phiJ = self.phi(J)
        if phiJ >= 1.0:
            return np.nan
        return phiJ + np.log(1.0 - phiJ) + self.chi * (phiJ**2)

    def H_double_prime(self, J: float) -> float:
        """Segunda derivada H''(J) numérica"""
        phiJ = self.phi(J)
        # evitar división por cero y dominios inválidos
        if J == 0 or phiJ >= 1.0:
            return np.nan
        return (phiJ**2) / J * (1.0 / (1.0 - phiJ) - 2.0 * self.chi)


# ===========================
# FUNCIONES MAESTRAS
# ===========================
def f_left(gel: gelsStability, Upsilon: float):
    """
    Determinante (real) para los rangos donde las cuatro raíces de la biquadrática
    en mu son reales. Se devuelve un float (o np.nan si falla).
    """
    a11 = gel.a11; a22 = gel.a22; b11 = gel.b11; b22 = gel.b22; a12 = gel.a12; b12 = gel.b12; L = gel.L

    # Valores intermedios
    b_Upsilon = -b11*b22 + b11*Upsilon - a11*a22 + a22*Upsilon + 4.0*(b12 - a12)**2
    a_Upsilon = a22*b11
    Delta_Upsilon = (b11**2 * b22**2 + a11**2 * a22**2 + 16.0 * (b12 - a12)**4 +
                    (a22 + b11)**2 * Upsilon**2 + 2.0 * a11 * a22 * b11 * b22 -
                    8.0 * a11 * a22 * (b12 - a12)**2 +
                    8.0 * (a22 + b11) * (b12 - a12)**2 * Upsilon -
                    2.0 * b11 * b22 * (a22 + b11) * Upsilon -
                    8.0 * b11 * b22 * (b12 - a12)**2 -
                    2.0 * a11 * a22 * (b11 + a22) * Upsilon -
                    4.0 * a22 * b11 * (a11 * b22 - a11 * Upsilon - b22 * Upsilon + Upsilon**2))

    if Delta_Upsilon < 0:
        return np.nan

    mu_1_arg = (-b_Upsilon - np.sqrt(Delta_Upsilon)) / (2.0 * a_Upsilon)
    mu_3_arg = (-b_Upsilon + np.sqrt(Delta_Upsilon)) / (2.0 * a_Upsilon)

    if mu_1_arg <= 0 or mu_3_arg <= 0:
        return np.nan

    mu_1 = np.sqrt(mu_1_arg)
    mu_3 = np.sqrt(mu_3_arg)

    denom1 = (b11 * mu_3**2 - a11 + Upsilon)
    if np.isclose(denom1, 0.0) or np.isclose(b12 - a12, 0.0):
        return np.nan
    p0 = (-2.0 * (b12 - a12)) / denom1
    s0 = (b11 * mu_1**2 - a11 + Upsilon) / (-2.0 * (b12 - a12))

    if np.isclose(s0, 0.0):
        return np.nan

    term1 = -(
        p0 * (b11 * mu_1**2 + 2.0 * b12 * s0) * (a22 * s0 + 2.0 * a12) +
        s0 * (a22 + 2.0 * a12 * p0) * (b11 * p0 * mu_3**2 + 2.0 * b12)
    )

    term2 = -(
        (mu_3**2 * p0 / mu_1) * s0 * (a22 + 2.0 * a12 * p0) * (b11 * mu_1**2 + 2.0 * b12 * s0)
        + mu_1 * (b11 * p0 * mu_3**2 + 2.0 * b12) * (a22 * s0 + 2.0 * a12)
    ) * np.sinh(mu_1 * L) * (np.sinh(mu_3 * L) / mu_3)

    term3 = (
        (a22 + 2.0 * a12 * p0) * (b11 * mu_1**2 + 2.0 * b12 * s0)
        + s0 * p0 * (b11 * p0 * mu_3**2 + 2.0 * b12) * (a22 * s0 + 2.0 * a12)
    ) * np.cosh(mu_1 * L) * np.cosh(mu_3 * L)

    result = term1 + term2 + term3
    return np.real(result) if np.isfinite(result) else np.nan


def f_left_left(gel: gelsStability, Upsilon: float):
    """
    Versión numérica de la función simbólica 'f_left_left'.
    Devuelve float (o np.nan en caso de condiciones inválidas).
    """
    a11 = gel.a11; a22 = gel.a22; b11 = gel.b11; b22 = gel.b22; a12 = gel.a12; b12 = gel.b12; L = gel.L

    a = a22 * b11
    b = (-b11*b22 + b11*Upsilon - a11*a22 + a22*Upsilon + 4.0*(b12 - a12)**2)
    c = (a11 - Upsilon)*(b22 - Upsilon)

    # Validaciones
    if a == 0.0:
        return np.nan
    inner = c / a
    # necesitamos inner >= 0 para sqrt; si no, devolvemos NaN (no aplicable en este rango)
    if inner < 0.0:
        return np.nan

    # calcular alpha y beta
    try:
        sqrt_inner = np.sqrt(inner)
        alpha_sq = (sqrt_inner - b / (2.0 * a)) / 2.0
        beta_sq  = (sqrt_inner + b / (2.0 * a)) / 2.0
        if alpha_sq < 0.0 or beta_sq < 0.0:
            return np.nan
        alpha = np.sqrt(alpha_sq)
        beta = np.sqrt(beta_sq)
    except Exception:
        return np.nan

    denom = -2.0 * (b12 - a12)
    if np.isclose(denom, 0.0) or np.isclose(b12 - a12, 0.0):
        return np.nan

    alpha_star = (b11*(alpha**2 - beta**2) - a11 + Upsilon) / denom
    beta_star  = -alpha * beta * b11 / (b12 - a12)

    rho1 = b11*(alpha**2 - beta**2) + 2.0 * b12 * alpha_star
    rho2 = -2.0 * b11 * alpha * beta - 2.0 * b12 * beta_star

    eta1 = a22*(alpha*alpha_star - beta*beta_star) + 2.0 * a12 * alpha
    eta2 = -a22*(alpha*beta_star + beta*alpha_star) - 2.0 * a12 * beta

    # Elementos de la matriz M (numéricos)
    m = np.exp(alpha*L)*(rho1*np.cos(beta*L) + rho2*np.sin(beta*L))
    n = np.exp(alpha*L)*(-rho2*np.cos(beta*L) + rho1*np.sin(beta*L))
    o = np.exp(-alpha*L)*(rho1*np.cos(beta*L) - rho2*np.sin(beta*L))
    p = np.exp(-alpha*L)*(rho2*np.cos(beta*L) + rho1*np.sin(beta*L))
    r = np.exp(alpha*L)*(eta1*np.cos(beta*L) + eta2*np.sin(beta*L))
    q = np.exp(alpha*L)*(-eta2*np.cos(beta*L) + eta1*np.sin(beta*L))
    t = -np.exp(-alpha*L)*(-eta1*np.cos(beta*L) + eta2*np.sin(beta*L))
    s = -np.exp(-alpha*L)*(-eta2*np.cos(beta*L) - eta1*np.sin(beta*L))

    # Ecuación final
    eq = (m*(q*(alpha*beta_star - beta*alpha_star) + s*(alpha*beta_star + beta*alpha_star) + 2.0*t*beta*beta_star)
        - n*(r*(alpha*beta_star - beta*alpha_star) + 2.0*s*alpha*alpha_star + t*(alpha*beta_star + beta*alpha_star))
        + o*(q*(alpha*beta_star + beta*alpha_star) - 2.0*r*beta*beta_star + s*(alpha*beta_star - beta*alpha_star))
        + p*(2.0*q*alpha*alpha_star - r*(alpha*beta_star + beta*alpha_star) + t*(beta*alpha_star - alpha*beta_star)))

    return np.real(eq) if np.isfinite(eq) else np.nan
