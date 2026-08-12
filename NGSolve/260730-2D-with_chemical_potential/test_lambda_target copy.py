from gels2d_ChemicalPotential import gel2d
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

L=90; d=1.62; 
phi0=0.2;
# phi0=1;
mu_bar_target = -0.002

dummy_mu_bar = mu_bar_target
gel = gel2d(L,d,phi0,dummy_mu_bar)

# lamb = 2.0
# mu_bar = gel.mu_fun(lamb)
# print(f'mu_bar = {mu_bar}')

lambda_start = 1.1
lambda_end = 6.0
num_points = 30
lambda_values = np.linspace(lambda_start,lambda_end,num_points)
mu_bar_values = [gel.mu_fun(lamb) for lamb in lambda_values]

plt.figure(figsize=(8, 5))
plt.plot(mu_bar_values, lambda_values, marker='o', linestyle='-', color='b')

plt.xlabel(r'$\bar{\mu}$')
plt.ylabel(r'$\lambda$')
plt.title(r'$\lambda$ vs. $\bar{\mu}$')
plt.grid(True)
plt.tight_layout()
plt.show()

# self.lambdaUniaxial = (scipy.optimize.fsolve(self.auxFunctionUniaxial, 2))[0]
def auxFunctionLambdaTarget(lamb):
    return gel.mu_fun(lamb) - mu_bar_target
lambda_target = (scipy.optimize.fsolve(auxFunctionLambdaTarget, 1.5))[0]
print(f'for phi0={phi0} and mu_bar_target={mu_bar_target}, lambda_target={lambda_target}')
print(f'gel.mu_fun(lambda_target)={gel.mu_fun(lambda_target)}')