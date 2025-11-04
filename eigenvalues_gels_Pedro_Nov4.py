import numpy as np
import math
from scipy.optimize import brentq

class gelsStability:
    # A class with the parameters of the gel
    # and the expression from the eigenvalue equation

    def __init__(self,                  
                 phi0=0.2,
                 gamma=1e-3,
                 chi=0.348,
                 epsilon=1.62/90.0,
                 L=1):
        # Initialize paremeters
        self.phi0 = phi0; self.gamma = gamma; self.chi = chi; self.epsilon = epsilon; self.L = L
        #
        H_prime = self.H_prime; H_double_prime = self.H_double_prime
        #
        # Compute homogeneous uniaxial swelling ratio
        f_aux_lambda_uniaxial = lambda _lambda : gamma*_lambda + H_prime(_lambda)
        print(f_aux_lambda_uniaxial(3.0))
        lam = brentq(f_aux_lambda_uniaxial, 1, 8)
        self.lam =lam;         
        # ===========================
        # Coeficientes a_ij y b_ij
        # ===========================        
        self.a11 = 4 * math.pi**2 * (gamma + lam**2 * H_double_prime(lam))
        self.a22 = gamma + H_double_prime(lam)
        self.b11 = epsilon**(-2) * gamma
        self.b22 = 4 * math.pi**2 * gamma * epsilon**2
        self.a12 = -math.pi * (lam * H_double_prime(lam) + H_prime(lam))
        self.b12 = -math.pi * H_prime(lam)
        gel=self; a11 = gel.a11; a22 = gel.a22; b11= gel.b11; b22=gel.b22; a12 = gel.a12; b12 = gel.b12;    
        # ===========================
        # Special values Upsilon_0 and Upsilon_1
        # ===========================
        # 1. cuadratic term (C2) - Coefficient of Upsilon^2
        C2 = (a22 - b11)**2
        # 2. Linear Coefficient (C1) - Coefficient of Upsilon
        term1_C1 = 8 * (a22 + b11) * (b12 - a12)**2
        term2_C1 = -2 * b11 * b22 * (a22 + b11)
        term3_C1 = -2 * a11 * a22 * (b11 + a22)
        term4_C1 = -4 * a22 * b11*(-a11-b22)
        C1 = term1_C1 + term2_C1 + term3_C1 + term4_C1
        # 3. Free Term (C0) - Constant term
        C0 = (b11**2 * b22**2 + a11**2 * a22**2 + 16 * (b12 - a12)**4 
                - 2 * a11 * a22 * b11 * b22 
                - 8 * a11 * a22 * (b12 - a12)**2
                - 8 * b11 * b22 * (b12 - a12)**2)
        # --- Calcular las raíces de P(Upsilon) ---
        discriminant = C1**2 - 4*C0*C2
        # print(f"Discriminant of P(Upsilon):{discriminant:.7f}")
        if discriminant < 0:
            root1 = (-C1 + complex(0, math.sqrt(-discriminant))) / (2*C2)
            root2 = (-C1 - complex(0, math.sqrt(-discriminant))) / (2*C2)
        else:
            root1 = (-C1 + math.sqrt(discriminant)) / (2*C2)
            root2 = (-C1 - math.sqrt(discriminant)) / (2*C2)
            self.Upsilon_0 = root2
            self.Upsilon_1 = root1        
        self.discriminant = discriminant
        # self.C1=C1; self.C2=C2;
        self.root1= root1; self.root2 = root2
        self.print_values()

    def print_values(self):
        print(f'lambda = {self.lam}, epsilon={self.epsilon:.4f}, phi0={self.phi0}, gamma={self.gamma}, chi = {self.chi}, L={self.L}')
        print(f'a11={self.a11}, a22={self.a22}, b11={self.b11}, b22={self.b22}')
        print(f'a12={self.a12}, b12={self.b12}')
        # print(f'C1 = {self.C1}, C2 = {self.C2}')
        discriminant = self.discriminant; root1=self.root1; root2=self.root2
        if discriminant < 0:
            print("No hay raíces reales, solo complejas.")
        # print(f"Raíces de P(Upsilon): {root1}, {root2}")
        else:
            print(f"Raíces de P(Upsilon): Upsilon_0={root2:.4e}, Upsilon_1={root1:.4e}")


    # ===========================
    # FUNCIONES AUXILIARES
    # ===========================
    def phi(self, J):
        phi0 = self.phi0
        return phi0 / J
    def H(self, J):
        phi0 = self.phi0; chi=self.chi; phi = self.phi 
        return (J - phi0) * np.log(1 - phi(J)) + phi0 * chi * (1 - phi(J))
    def H_prime(self, J):
        chi=self.chi; phi=self.phi
        return phi(J) + np.log(1 - phi(J)) + chi * phi(J)**2
    def H_double_prime(self, J):
        chi=self.chi; phi=self.phi
        return (phi(J))**2 / J * (1 / (1 - phi(J)) - 2 * chi)


# ===========================
# FUNCIONES MAESTRAS
# ===========================
# _f_left has the expression for the determinant
# of the system that applies both to:
# the range Upsilon < Upsilon_0
# and the range Upsilon_1 < Upsilon < b_{22}
# In both of those ranges, the four roots
# of the biquadratic equation for \mu are real
def f_left(gel, Upsilon):
    a11 = gel.a11; a22 = gel.a22; b11= gel.b11; b22=gel.b22; a12 = gel.a12; b12 = gel.b12; L=gel.L;
    #
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
