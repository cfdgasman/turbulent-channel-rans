import numpy as np
import pytest

from channel import dean_cf, k_omega, mixing_length, reichardt


@pytest.fixture(scope="module")
def kw550():
    return k_omega(550)


def test_total_stress_balance_mixing_length():
    p = mixing_length(1000)
    dudy = np.gradient(p.U, p.y)
    assert np.allclose((1 / 1000 + p.nu_t) * dudy, 1 - p.y, atol=2e-3)


def test_k_omega_converges_and_balances_stress(kw550):
    assert kw550.iterations < 50_000
    dudy = np.gradient(kw550.U, kw550.y)
    assert np.abs((1 / 550 + kw550.nu_t) * dudy - (1 - kw550.y)).max() < 1e-2


def test_viscous_sublayer(kw550):
    m = kw550.y_plus < 1
    assert np.allclose(kw550.U[m], kw550.y_plus[m], rtol=0.02, atol=1e-3)


def test_log_law_intercept(kw550):
    m = (kw550.y_plus > 50) & (kw550.y < 0.2)
    B = np.mean(kw550.U[m] - np.log(kw550.y_plus[m]) / 0.41)
    assert 4.5 < B < 5.5


def test_skin_friction_vs_dean(kw550):
    assert kw550.cf == pytest.approx(dean_cf(kw550.re_bulk), rel=0.05)


def test_reichardt_limits():
    assert reichardt(0.5) == pytest.approx(0.5, rel=0.02)
    assert reichardt(1000) == pytest.approx(np.log(1000) / 0.41 + 7.8 + np.log(0.41) / 0.41, abs=0.02)
