"""Turbulent channel flow: mixing-length and k-omega RANS vs the law of the wall and Dean's correlation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel import dean_cf, k_omega, mixing_length, reichardt
from channel.rans import KAPPA

RE_TAUS = [180, 550, 1000, 2000]


def log_intercept(p):
    m = (p.y_plus > 50) & (p.y < 0.2)
    return np.mean(p.U[m] - np.log(p.y_plus[m]) / KAPPA) if m.any() else np.nan


def main():
    ml = {re: mixing_length(re) for re in RE_TAUS}
    kw = {re: k_omega(re) for re in RE_TAUS}

    print("| Re_tau | model | iterations | U_b+ | Re_b | Cf | Cf Dean | error | log-law B |")
    print("|---|---|---|---|---|---|---|---|---|")
    for re in RE_TAUS:
        for name, p in (("mixing length", ml[re]), ("k-omega", kw[re])):
            d = dean_cf(p.re_bulk)
            print(f"| {re} | {name} | {p.iterations or '-'} | {p.bulk_velocity:.2f} | {p.re_bulk:,.0f} | "
                  f"{p.cf:.5f} | {d:.5f} | {100 * (p.cf / d - 1):+.1f} % | {log_intercept(p):.2f} |")

    # --- velocity profiles in wall units
    fig, ax = plt.subplots(figsize=(7, 4.8))
    yp = np.logspace(-1, np.log10(2500), 300)
    ax.semilogx(yp, reichardt(yp), "k-", lw=2.5, alpha=0.25, label="Reichardt law of the wall")
    ax.semilogx(yp[yp < 12], yp[yp < 12], "k:", lw=1, label="u⁺ = y⁺")
    ax.semilogx(yp[yp > 20], np.log(yp[yp > 20]) / KAPPA + 5.0, "k--", lw=1, label="u⁺ = ln(y⁺)/0.41 + 5.0")
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(RE_TAUS)))
    for c, re in zip(colors, RE_TAUS):
        ax.semilogx(kw[re].y_plus[1:], kw[re].U[1:], "-", color=c, lw=1.8, label=f"k-ω, Re_τ = {re}")
        ax.semilogx(ml[re].y_plus[1:], ml[re].U[1:], "--", color=c, lw=1.2)
    ax.set(xlabel="y⁺", ylabel="U⁺", xlim=(0.1, 2500), ylim=(0, 28),
           title="Mean velocity (solid: k-ω, dashed: mixing length)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("docs/velocity_profiles.png", dpi=130)

    # --- skin friction
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    reb = np.logspace(np.log10(4e3), np.log10(1.2e5), 100)
    ax.loglog(reb, dean_cf(reb), "k-", label="Dean (1978): 0.073 Re_b^(-1/4)")
    ax.loglog([ml[r].re_bulk for r in RE_TAUS], [ml[r].cf for r in RE_TAUS], "s", mfc="none", label="mixing length")
    ax.loglog([kw[r].re_bulk for r in RE_TAUS], [kw[r].cf for r in RE_TAUS], "o", label="k-ω (Wilcox 1988)")
    ax.set(xlabel="Re_b = U_b 2δ / ν", ylabel="C_f", title="Skin friction")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/skin_friction.png", dpi=130)

    # --- turbulence quantities
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, re in zip(colors, RE_TAUS):
        p = kw[re]
        a1.semilogx(p.y_plus[1:], p.k[1:], color=c, label=f"Re_τ = {re}")
        a2.plot(p.y, p.nu_t * re, color=c, label=f"k-ω, Re_τ = {re}")
        a2.plot(ml[re].y, ml[re].nu_t * re, "--", color=c, lw=1)
    a1.axhline(1 / np.sqrt(0.09), color="k", ls=":", lw=1, label="log-layer k⁺ = 1/√β*")
    a1.set(xlabel="y⁺", ylabel="k⁺", title="Turbulent kinetic energy (k-ω)")
    a2.set(xlabel="y / δ", ylabel="ν_t / ν", title="Eddy viscosity (dashed: mixing length)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/turbulence_quantities.png", dpi=130)


if __name__ == "__main__":
    main()
