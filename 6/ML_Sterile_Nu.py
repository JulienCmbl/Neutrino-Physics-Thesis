"""
Fast Sterile Neutrino Mixing Angle Search
==========================================
Uses the averaged oscillation probability directly from PMNS matrix elements:

    P(νe → νs) = Σᵢ |U_ei|² |U_si|²    (i = 1,2,3,4)

Three strategies:
  1. Latin Hypercube Scan   — maps the full feasible region
  2. Bayesian Opt (Optuna)  — smart sampling, fast convergence
  3. PyTorch Gradient Descent — gradient-based, finds a precise minimum

Constraint: P(νe → νs) < P_STERILE_THRESHOLD = 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


DCP = 0.0

squared_s12 = 0.307;  s12 = np.sqrt(squared_s12);  c12 = np.sqrt(1 - squared_s12)
squared_s13 = 0.0219; s13 = np.sqrt(squared_s13);  c13 = np.sqrt(1 - squared_s13)
squared_s23 = 0.558;  s23 = np.sqrt(squared_s23);  c23 = np.sqrt(1 - squared_s23)


ANGLE_MIN             = 0.0
ANGLE_MAX             = np.pi / (4 * 3)      # 45° — physically motivated upper bound
P_STERILE_THRESHOLD   = 0.1


def build_PMNS(t14, t24, t34):
    eCP  = np.exp( 1j * DCP)
    enCP = np.exp(-1j * DCP)

    U_pmns = np.array([
        [c12*c13,                            s12*c13,                            s13*enCP,  0],
        [-s12*c23 - c12*s23*s13*eCP,          c12*c23 - s12*s23*s13*eCP,          s23*c13,  0],
        [ s12*s23 - c12*c23*s13*eCP,         -c12*s23 - s12*c23*s13*eCP,          c23*c13,  0],
        [0,                                   0,                                   0,        1]
    ], dtype=complex)

    s14_, c14_ = np.sin(t14), np.cos(t14)
    s24_, c24_ = np.sin(t24), np.cos(t24)
    s34_, c34_ = np.sin(t34), np.cos(t34)

    R34 = np.array([[1,0,0,0],[0,1,0,0],[0,0,c34_,s34_],[0,0,-s34_,c34_]])
    R24 = np.array([[1,0,0,0],[0,c24_,0,s24_],[0,0,1,0],[0,-s24_,0,c24_]])
    R14 = np.array([[c14_,0,0,s14_],[0,1,0,0],[0,0,1,0],[-s14_,0,0,c14_]])

    return R34 @ R24 @ R14 @ U_pmns


def averaged_prob_e_to_s(t14, t24, t34):
  
    U = build_PMNS(t14, t24, t34)
    U_ei_sq = np.abs(U[0, :])**2    # |U_e1|², |U_e2|², |U_e3|², |U_e4|²
    U_si_sq = np.abs(U[3, :])**2    # |U_s1|², |U_s2|², |U_s3|², |U_s4|²
    return float(np.sum(U_ei_sq * U_si_sq))


def print_matrix_elements(t14, t24, t34):
  
    U = build_PMNS(t14, t24, t34)
    U_sq = np.abs(U)**2
    flavours = ["νe", "νμ", "ντ", "νs"]
    mass     = ["ν₁", "ν₂", "ν₃", "ν₄"]
    print(f"\n  |U|² matrix  (θ₁₄={np.degrees(t14):.2f}°, "
          f"θ₂₄={np.degrees(t24):.2f}°, θ₃₄={np.degrees(t34):.2f}°)")
    print("        " + "   ".join(f"{m:>6}" for m in mass))
    for i, fl in enumerate(flavours):
        row = "   ".join(f"{U_sq[i,j]:.4f}" for j in range(4))
        print(f"  {fl}  [ {row} ]")
    p = averaged_prob_e_to_s(t14, t24, t34)
    print(f"\n  P(νe→νs) = Σᵢ |U_ei|²|U_si|² = {p:.6f}")



print("="*60)
print("SANITY CHECK — original angles from your simulation")
print("="*60)
T14_orig = 0.0872665   # ~5°
T24_orig = 0.122173    # ~7°
T34_orig = 0.174533    # ~10°
print_matrix_elements(T14_orig, T24_orig, T34_orig)
p_orig = averaged_prob_e_to_s(T14_orig, T24_orig, T34_orig)
if p_orig > P_STERILE_THRESHOLD:
    print(f"\n  Original angles give P_s = {p_orig:.4f} > {P_STERILE_THRESHOLD} → search needed")
else:
    print(f"\n  Original angles already satisfy constraint: P_s = {p_orig:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — Latin Hypercube Scan
# ──────────────────────────────────────────────────────────────────────────────
def latin_hypercube_scan(n_samples=5000, seed=42):
    print("\n" + "="*60)
    print("STRATEGY 1 — Latin Hypercube Scan")
    print(f"  {n_samples} samples in [0°, 45°]³")
    print("="*60)

    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, 3))
    for d in range(3):
        perm = rng.permutation(n_samples)
        samples[:, d] = (perm + rng.random(n_samples)) / n_samples
    angles  = ANGLE_MIN + samples * (ANGLE_MAX - ANGLE_MIN)

    results = np.array([averaged_prob_e_to_s(a[0], a[1], a[2]) for a in angles])
    feasible = results < P_STERILE_THRESHOLD

    print(f"\n  Feasible: {feasible.sum()} / {n_samples}  "
          f"({100*feasible.mean():.1f}%)")

    if feasible.sum() > 0:
        fa = angles[feasible]
        print(f"\n  Feasible angle bounds (5th–95th percentile):")
        for j, name in enumerate(["θ₁₄", "θ₂₄", "θ₃₄"]):
            p5  = np.degrees(np.percentile(fa[:,j],  5))
            p50 = np.degrees(np.percentile(fa[:,j], 50))
            p95 = np.degrees(np.percentile(fa[:,j], 95))
            print(f"    {name}:  [{p5:.2f}°,  {p95:.2f}°]   median {p50:.2f}°")

    return angles, results, feasible


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — Bayesian Optimisation (Optuna)
# ──────────────────────────────────────────────────────────────────────────────
def bayesian_optimisation(n_trials=500):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [!] Install optuna:  pip install optuna")
        return None, None, None

    print("\n" + "="*60)
    print("STRATEGY 2 — Bayesian Optimisation (Optuna TPE)")
    print(f"  {n_trials} trials")
    print("="*60)

    trial_angles  = []
    trial_results = []

    def objective(trial):
        t14 = trial.suggest_float("theta14", ANGLE_MIN, ANGLE_MAX)
        t24 = trial.suggest_float("theta24", ANGLE_MIN, ANGLE_MAX)
        t34 = trial.suggest_float("theta34", ANGLE_MIN, ANGLE_MAX)
        p   = averaged_prob_e_to_s(t14, t24, t34)
        trial_angles.append([t14, t24, t34])
        trial_results.append(p)
        return p

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    angles  = np.array(trial_angles)
    results = np.array(trial_results)
    feasible = results < P_STERILE_THRESHOLD

    best = study.best_params
    print(f"\n  Best:  P_s = {study.best_value:.6f}")
    print(f"    θ₁₄ = {np.degrees(best['theta14']):.3f}°")
    print(f"    θ₂₄ = {np.degrees(best['theta24']):.3f}°")
    print(f"    θ₃₄ = {np.degrees(best['theta34']):.3f}°")

    if feasible.sum() > 0:
        fa = angles[feasible]
        print(f"\n  Feasible angle bounds (5th–95th percentile):")
        for j, name in enumerate(["θ₁₄", "θ₂₄", "θ₃₄"]):
            p5  = np.degrees(np.percentile(fa[:,j],  5))
            p50 = np.degrees(np.percentile(fa[:,j], 50))
            p95 = np.degrees(np.percentile(fa[:,j], 95))
            print(f"    {name}:  [{p5:.2f}°,  {p95:.2f}°]   median {p50:.2f}°")

    return study, angles, results


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — PyTorch Gradient Descent
# ──────────────────────────────────────────────────────────────────────────────
def pytorch_gradient_descent(n_restarts=20, n_steps=200, lr=0.05):
    try:
        import torch
    except ImportError:
        print("  [!] Install torch:  pip install torch")
        return None

    print("\n" + "="*60)
    print("STRATEGY 3 — PyTorch Gradient Descent")
    print(f"  {n_restarts} restarts × {n_steps} steps   lr={lr}")
    print("="*60)

    def build_PMNS_torch(t14, t24, t34):
        eCP  = torch.tensor(np.exp( 1j*DCP), dtype=torch.complex128)
        enCP = torch.tensor(np.exp(-1j*DCP), dtype=torch.complex128)

        # Cast real trig values to complex for matrix assembly
        def rc(x): return x.to(torch.complex128)

        s12_ = torch.tensor(s12); c12_ = torch.tensor(c12)
        s13_ = torch.tensor(s13); c13_ = torch.tensor(c13)
        s23_ = torch.tensor(s23); c23_ = torch.tensor(c23)

        U_pmns = torch.zeros(4, 4, dtype=torch.complex128)
        U_pmns[0,0] =  rc(c12_*c13_)
        U_pmns[0,1] =  rc(s12_*c13_)
        U_pmns[0,2] =  rc(s13_)*enCP
        U_pmns[1,0] =  rc(-s12_*c23_ - c12_*s23_*s13_)*eCP
        U_pmns[1,1] =  rc( c12_*c23_ - s12_*s23_*s13_)*eCP
        U_pmns[1,2] =  rc(s23_*c13_)
        U_pmns[2,0] =  rc( s12_*s23_ - c12_*c23_*s13_)*eCP
        U_pmns[2,1] =  rc(-c12_*s23_ - s12_*c23_*s13_)*eCP
        U_pmns[2,2] =  rc(c23_*c13_)
        U_pmns[3,3] =  torch.tensor(1.0, dtype=torch.complex128)

        s14_ = torch.sin(t14); c14_ = torch.cos(t14)
        s24_ = torch.sin(t24); c24_ = torch.cos(t24)
        s34_ = torch.sin(t34); c34_ = torch.cos(t34)

        z = torch.zeros(1, dtype=torch.complex128).squeeze()
        o = torch.ones (1, dtype=torch.complex128).squeeze()
        def cc(x): return x.to(torch.complex128)

        R34 = torch.stack([
            torch.stack([o,z,z,z]),
            torch.stack([z,o,z,z]),
            torch.stack([z,z,cc(c34_),cc(s34_)]),
            torch.stack([z,z,-cc(s34_),cc(c34_)])
        ])
        R24 = torch.stack([
            torch.stack([o,z,z,z]),
            torch.stack([z,cc(c24_),z,cc(s24_)]),
            torch.stack([z,z,o,z]),
            torch.stack([z,-cc(s24_),z,cc(c24_)])
        ])
        R14 = torch.stack([
            torch.stack([cc(c14_),z,z,cc(s14_)]),
            torch.stack([z,o,z,z]),
            torch.stack([z,z,o,z]),
            torch.stack([-cc(s14_),z,z,cc(c14_)])
        ])
        return R34 @ R24 @ R14 @ U_pmns

    def prob_torch(t14, t24, t34):
        U     = build_PMNS_torch(t14, t24, t34)
        Uei_sq = torch.abs(U[0,:])**2   # νe row
        Usi_sq = torch.abs(U[3,:])**2   # νs row
        return torch.sum(Uei_sq * Usi_sq)

    rng      = np.random.default_rng(0)
    all_runs = []
    best_p   = 1.0
    best_angles = None

    for restart in range(n_restarts):
        init = rng.uniform(ANGLE_MIN, ANGLE_MAX, 3)
        t14 = torch.tensor(init[0], dtype=torch.float64, requires_grad=True)
        t24 = torch.tensor(init[1], dtype=torch.float64, requires_grad=True)
        t34 = torch.tensor(init[2], dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.Adam([t14, t24, t34], lr=lr)
        history   = []

        for step in range(n_steps):
            optimizer.zero_grad()
            t14c = torch.clamp(t14, ANGLE_MIN, ANGLE_MAX)
            t24c = torch.clamp(t24, ANGLE_MIN, ANGLE_MAX)
            t34c = torch.clamp(t34, ANGLE_MIN, ANGLE_MAX)
            loss = prob_torch(t14c, t24c, t34c)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        final_p = history[-1]
        a = (
            np.degrees(float(np.clip(t14.item(), ANGLE_MIN, ANGLE_MAX))),
            np.degrees(float(np.clip(t24.item(), ANGLE_MIN, ANGLE_MAX))),
            np.degrees(float(np.clip(t34.item(), ANGLE_MIN, ANGLE_MAX))),
        )
        all_runs.append({
            "restart": restart+1, "p_sterile": final_p,
            "theta14": a[0], "theta24": a[1], "theta34": a[2],
            "history": history
        })

        marker = "✓" if final_p < P_STERILE_THRESHOLD else "✗"
        print(f"  [{marker}] Restart {restart+1:2d}: P_s={final_p:.5f}  "
              f"θ₁₄={a[0]:.2f}°  θ₂₄={a[1]:.2f}°  θ₃₄={a[2]:.2f}°")

        if final_p < best_p:
            best_p      = final_p
            best_angles = a

    print(f"\n  Best:  P_s = {best_p:.6f}")
    print(f"    θ₁₄ = {best_angles[0]:.3f}°")
    print(f"    θ₂₄ = {best_angles[1]:.3f}°")
    print(f"    θ₃₄ = {best_angles[2]:.3f}°")

    return all_runs


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def plot_2d_projections(angles, results, title, filename):
    feasible = results < P_STERILE_THRESHOLD
    pairs = [(0,1,"θ₁₄ (°)","θ₂₄ (°)"),
             (0,2,"θ₁₄ (°)","θ₃₄ (°)"),
             (1,2,"θ₂₄ (°)","θ₃₄ (°)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        sc = ax.scatter(np.degrees(angles[:,i]), np.degrees(angles[:,j]),
                        c=results, cmap="RdYlGn_r", vmin=0, vmax=0.3,
                        s=8, alpha=0.6)
        if feasible.sum() > 0:
            ax.scatter(np.degrees(angles[feasible,i]),
                       np.degrees(angles[feasible,j]),
                       edgecolors="royalblue", facecolors="none",
                       s=25, linewidths=0.8,
                       label=f"P_s < {P_STERILE_THRESHOLD}")
            ax.legend(fontsize=8)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
    plt.colorbar(sc, ax=axes[-1], label="P(νe→νs)")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {filename}")


def plot_torch_convergence(all_runs):
    if all_runs is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for run in all_runs:
        color = "green" if run["p_sterile"] < P_STERILE_THRESHOLD else "tomato"
        ax.plot(run["history"], alpha=0.7, linewidth=1.2, color=color)
    ax.axhline(P_STERILE_THRESHOLD, color="black", ls="--",
               label=f"Threshold {P_STERILE_THRESHOLD}")
    ax.set_xlabel("Step"); ax.set_ylabel("P(νe→νs)")
    ax.set_title("PyTorch — Convergence"); ax.legend()

    ax2 = axes[1]
    ps  = [r["p_sterile"] for r in all_runs]
    idx = [r["restart"]   for r in all_runs]
    bar_colors = ["green" if p < P_STERILE_THRESHOLD else "tomato" for p in ps]
    ax2.bar(idx, ps, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax2.axhline(P_STERILE_THRESHOLD, color="black", ls="--",
                label=f"Threshold {P_STERILE_THRESHOLD}")
    ax2.set_xlabel("Restart"); ax2.set_ylabel("Final P(νe→νs)")
    ax2.set_title("PyTorch — Final P_s per Restart"); ax2.legend()

    plt.tight_layout()
    plt.savefig("pytorch_fast_results.pdf", bbox_inches="tight")
    plt.show()
    print("  Saved: pytorch_fast_results.pdf")


def print_summary(lhs_angles, lhs_results,
                  opt_angles, opt_results,
                  torch_runs):
    print("\n" + "="*60)
    print("FINAL SUMMARY — P(νe→νs) < 0.1  bounds")
    print("="*60)

    for label, angles, results in [
        ("LHS",    lhs_angles, lhs_results),
        ("Optuna", opt_angles, opt_results),
    ]:
        if angles is None:
            continue
        feasible = results < P_STERILE_THRESHOLD
        if feasible.sum() == 0:
            print(f"\n  {label}: no feasible samples found")
            continue
        fa = angles[feasible]
        print(f"\n  {label}  ({feasible.sum()} feasible / {len(results)} total):")
        for j, name in enumerate(["θ₁₄", "θ₂₄", "θ₃₄"]):
            p5  = np.degrees(np.percentile(fa[:,j],  5))
            p50 = np.degrees(np.percentile(fa[:,j], 50))
            p95 = np.degrees(np.percentile(fa[:,j], 95))
            print(f"    {name}:  5th–95th = [{p5:.2f}°, {p95:.2f}°]   "
                  f"median {p50:.2f}°")

    if torch_runs is not None:
        best = min(torch_runs, key=lambda r: r["p_sterile"])
        print(f"\n  PyTorch best single solution:  P_s = {best['p_sterile']:.6f}")
        print(f"    θ₁₄ = {best['theta14']:.3f}°")
        print(f"    θ₂₄ = {best['theta24']:.3f}°")
        print(f"    θ₃₄ = {best['theta34']:.3f}°")
        print_matrix_elements(
            np.radians(best["theta14"]),
            np.radians(best["theta24"]),
            np.radians(best["theta34"])
        )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Strategy 1 — LHS  (5000 samples runs in < 1 second)
    lhs_angles, lhs_results, lhs_feasible = latin_hypercube_scan(n_samples=5000)
    plot_2d_projections(lhs_angles, lhs_results,
                        "LHS Scan — P(νe→νs) over Sterile Angle Space",
                        "lhs_fast_scan.pdf")

    # Strategy 2 — Optuna  (500 trials runs in a few seconds)
    study, opt_angles, opt_results = bayesian_optimisation(n_trials=500)
    if opt_angles is not None:
        plot_2d_projections(opt_angles, opt_results,
                            "Optuna — Explored Sterile Angle Space",
                            "optuna_fast_scan.pdf")

    # Strategy 3 — PyTorch  (20 restarts × 200 steps, each step is one matmul)
    torch_runs = pytorch_gradient_descent(n_restarts=20, n_steps=200, lr=0.05)
    plot_torch_convergence(torch_runs)

    # Final summary
    print_summary(lhs_angles, lhs_results,
                  opt_angles, opt_results,
                  torch_runs)
