import numpy as np

# ============================================================
# Parámetros del material (ajústalos a tu caso)
# ============================================================
Nv = 0.001
w = 0.4

# ============================================================
# Ecuación (3.6) Kang & Huang (2010)
# Devuelve mu/(k_B T)
# ============================================================
def mu_fun(lambda_val, Nv, w):
    return (
        np.log(1.0 - 1.0 / lambda_val)
        + 1.0 / lambda_val
        + w / (lambda_val ** 2)
        + Nv * (lambda_val - 1.0 / lambda_val)
    )

# ============================================================
# Generador de path en lambda → mu
# ============================================================
def build_mu_path(lambda_start=1.1, lambda_target=2.0, n_steps=16):
    lambdas = np.linspace(lambda_start, lambda_target, n_steps)
    mus = np.array([mu_fun(l, Nv, w) for l in lambdas])
    return lambdas, mus

# ============================================================
# NUEVO SOLVER incremental en μ
# ============================================================
def Solve_incremental_mu(gel, SolveNonlinearMinProblem,
                         lambda_start=1.1,
                         lambda_target=2.0,
                         n_steps=16):

    lambdas, mus = build_mu_path(lambda_start, lambda_target, n_steps)

    results = []

    print("\n=== Incremental solve en μ ===\n")

    for i, (lambda_val, mu) in enumerate(zip(lambdas, mus)):

        print(f"[Paso {i+1}/{n_steps}] λ = {lambda_val:.4f}, μ = {mu:.6f}")

        # ----------------------------------------------------
        # 🔴 CAMBIO CLAVE:
        # ahora controlas μ en vez de γ
        # ----------------------------------------------------
        gel.mu_bar = mu

        # ----------------------------------------------------
        # (Opcional pero MUY recomendable)
        # usar lambda como predictor inicial
        # ----------------------------------------------------
        if hasattr(gel, "set_lambda_predictor"):
            gel.set_lambda_predictor(lambda_val)

        # ----------------------------------------------------
        # Resolver problema no lineal
        # ----------------------------------------------------
        sol = SolveNonlinearMinProblem(gel)

        results.append({
            "lambda": lambda_val,
            "mu": mu,
            "solution": sol
        })

    return results