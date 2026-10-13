"""Fully developed turbulent channel flow with RANS turbulence models.

Everything is in wall units: half-height delta = 1, friction velocity u_tau = 1,
so nu = 1 / Re_tau and the driving pressure gradient is dp/dx = -1. The
mean-momentum balance is

    d/dy [ (nu + nu_t) dU/dy ] = -1,     U(0) = 0,  dU/dy(1) = 0.

Models
------
* ``mixing_length``: Prandtl mixing length with van Driest damping (A+ = 26) and
  the Nikuradse outer-layer length scale. The total-stress balance
  (nu + l^2 |U'|) U' = 1 - y is solved pointwise (a quadratic) and integrated.
* ``k_omega``: Wilcox (1988) k-omega model, solved on a wall-clustered grid by
  Picard iteration with implicit tridiagonal solves. The wall value of omega
  follows Menter: omega_w = 10 * 6 nu / (beta * y1^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.integrate import cumulative_trapezoid

KAPPA = 0.41


@dataclass
class Profile:
    re_tau: float
    y: np.ndarray  # y / delta, 0 (wall) .. 1 (centreline)
    U: np.ndarray  # U / u_tau
    nu_t: np.ndarray  # nu_t / (u_tau delta)
    k: np.ndarray | None = None
    omega: np.ndarray | None = None
    iterations: int = 0

    @property
    def y_plus(self):
        return self.y * self.re_tau

    @property
    def bulk_velocity(self):
        return np.trapezoid(self.U, self.y)  # U_b+ (half-channel average)

    @property
    def re_bulk(self):
        """Bulk Reynolds number based on the full channel height 2 delta."""
        return 2.0 * self.bulk_velocity * self.re_tau

    @property
    def cf(self):
        return 2.0 / self.bulk_velocity**2


def grid(re_tau, n=200, y1_plus=0.2):
    """Tanh-clustered grid on [0, 1] with the first point at y+ ~ y1_plus."""
    target = y1_plus / re_tau
    lo, hi = 0.1, 20.0
    for _ in range(100):  # bisection for the stretching factor
        g = 0.5 * (lo + hi)
        y1 = 1 - np.tanh(g * (1 - 1 / n)) / np.tanh(g)
        lo, hi = (g, hi) if y1 > target else (lo, g)
    eta = np.linspace(0, 1, n + 1)
    return 1 - np.tanh(g * (1 - eta)) / np.tanh(g)


def reichardt(y_plus):
    """Reichardt (1951) composite law of the wall."""
    return np.log(1 + KAPPA * y_plus) / KAPPA + 7.8 * (
        1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3)
    )


def dean_cf(re_b):
    """Dean (1978) correlation for channel skin friction, Re_b based on 2 delta."""
    return 0.073 * re_b ** -0.25


# ---------------------------------------------------------------- mixing length


def mixing_length(re_tau, n=400, a_plus=26.0):
    nu = 1.0 / re_tau
    y = grid(re_tau, n)
    eta = 1 - y
    l_outer = 0.14 - 0.08 * eta**2 - 0.06 * eta**4  # Nikuradse, l / delta
    l = l_outer * (1 - np.exp(-y * re_tau / a_plus))
    tau = 1 - y
    dudy = 2 * tau / (nu + np.sqrt(nu**2 + 4 * l**2 * tau))
    U = cumulative_trapezoid(dudy, y, initial=0.0)
    return Profile(re_tau, y, U, l**2 * np.abs(dudy))


# ---------------------------------------------------------------- k-omega


ALPHA, BETA, BETA_S, SIGMA, SIGMA_S = 5 / 9, 3 / 40, 9 / 100, 0.5, 0.5


def _diffusion_system(y, gamma_face, sink, src, wall_value):
    """Assemble -d/dy(Gamma dphi/dy) + sink * phi = src with phi(0) = wall_value and
    symmetry at y = 1. Returns the banded matrix and right-hand side."""
    n = len(y) - 1
    ab = np.zeros((3, n + 1))
    rhs = src.copy()
    dy = np.diff(y)
    # interior nodes 1..n-1
    vol = 0.5 * (dy[1:] + dy[:-1])
    aw = gamma_face[:-1] / dy[:-1] / vol
    ae = gamma_face[1:] / dy[1:] / vol
    ab[1, 1:n] = aw + ae + sink[1:n]
    ab[0, 2 : n + 1] = -ae  # super-diagonal
    ab[2, 0 : n - 1] = -aw  # sub-diagonal
    # symmetry node n: half cell, zero flux at the centreline
    vol_n = 0.5 * dy[-1]
    aw_n = gamma_face[-1] / dy[-1] / vol_n
    ab[1, n] = aw_n + sink[n]
    ab[2, n - 1] = -aw_n
    # wall node: Dirichlet
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    rhs[0] = wall_value
    return ab, rhs


def k_omega(re_tau, n=200, tol=1e-9, max_iter=50_000, relax=0.1):
    nu = 1.0 / re_tau
    y = grid(re_tau, n)
    ml = mixing_length(re_tau)  # initial guess
    U = np.interp(y, ml.y, ml.U)
    nu_t = np.interp(y, ml.y, ml.nu_t)
    k = np.full_like(y, 1.0)
    k[0] = 0.0
    omega = np.maximum(k / np.maximum(nu_t, 1e-3 * nu), 1.0)
    omega_wall = 10 * 6 * nu / (BETA * y[1] ** 2)
    omega[0] = omega_wall
    zero = np.zeros_like(y)

    face = lambda a: 0.5 * (a[1:] + a[:-1])  # noqa: E731
    it = 0
    for it in range(1, max_iter + 1):
        U_old, k_old, w_old = U.copy(), k.copy(), omega.copy()

        # momentum: -d/dy((nu + nu_t) dU/dy) = 1
        ab, rhs = _diffusion_system(y, nu + face(nu_t), zero, np.ones_like(y), 0.0)
        U = solve_banded((1, 1), ab, rhs)

        dudy = np.gradient(U, y)
        S2 = dudy**2

        # k: -d/dy((nu + s* nu_t) dk/dy) + beta* omega k = nu_t S^2
        ab, rhs = _diffusion_system(y, nu + SIGMA_S * face(nu_t), BETA_S * omega, nu_t * S2, 0.0)
        k = np.maximum(solve_banded((1, 1), ab, rhs), 1e-14)

        # omega: -d/dy((nu + s nu_t) domega/dy) + beta omega_old omega = alpha S^2
        ab, rhs = _diffusion_system(y, nu + SIGMA * face(nu_t), BETA * omega, ALPHA * S2, omega_wall)
        omega = np.maximum(solve_banded((1, 1), ab, rhs), 1e-10)

        nu_t = (1 - relax) * nu_t + relax * k / omega
        nu_t[0] = 0.0

        change = max(
            np.abs(U - U_old).max() / np.abs(U).max(),
            np.abs(k - k_old).max() / np.abs(k).max(),
            np.abs(omega[1:] - w_old[1:]).max() / np.abs(omega[1:]).max(),
        )
        if change < tol:
            break
    return Profile(re_tau, y, U, nu_t, k, omega, it)
