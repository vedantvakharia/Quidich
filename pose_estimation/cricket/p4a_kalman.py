"""Singer-model Kalman filter on 2D ground plane for P4a global tracking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov


@dataclass(frozen=True)
class RoleParams:
    alpha: float          # maneuver frequency (1/s, higher = more maneuverable)
    sigma_a: float        # acceleration noise std (m/s^2)
    measurement_noise: float  # position measurement std (m)


# Fix 2.1: striker has non-trivial Q because they sprint between wickets.
ROLE_PARAMS: dict[str, RoleParams] = {
    "bowler":       RoleParams(alpha=2.0,  sigma_a=3.0,  measurement_noise=0.3),
    "striker":      RoleParams(alpha=1.5,  sigma_a=2.5,  measurement_noise=0.3),
    "non_striker":  RoleParams(alpha=0.5,  sigma_a=1.0,  measurement_noise=0.3),
    "wicketkeeper": RoleParams(alpha=0.3,  sigma_a=0.5,  measurement_noise=0.2),
    "umpire":       RoleParams(alpha=0.2,  sigma_a=0.3,  measurement_noise=0.2),
    "fielder":      RoleParams(alpha=1.0,  sigma_a=2.0,  measurement_noise=0.4),
    "unknown":      RoleParams(alpha=1.0,  sigma_a=2.0,  measurement_noise=0.4),
}

_CHI2_95_2DOF = 5.991  # 95th percentile chi-squared, 2 DOF


def _singer_dynamics(alpha: float, sigma_a: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute (F_d, Q_d) for Singer model. State: [x, y, vx, vy, ax, ay]."""
    n = 6
    Fc = np.zeros((n, n))
    Fc[0, 2] = 1.0; Fc[1, 3] = 1.0  # pos <- vel
    Fc[2, 4] = 1.0; Fc[3, 5] = 1.0  # vel <- acc
    Fc[4, 4] = -alpha; Fc[5, 5] = -alpha

    F_d = expm(Fc * dt)

    # Van Loan method for Q_d
    G = np.zeros((n, 2))
    G[4, 0] = 1.0; G[5, 1] = 1.0
    Q_c = np.eye(2) * (sigma_a ** 2)
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -Fc
    M[:n, n:] = G @ Q_c @ G.T
    M[n:, n:] = Fc.T
    expM = expm(M * dt)
    Q_d = expM[n:, n:].T @ expM[:n, n:]
    Q_d = 0.5 * (Q_d + Q_d.T)  # symmetrize
    return F_d, Q_d


class SingerGroundKalman:
    """Singer-model Kalman filter on 2D ground plane. Mahalanobis gating, proper P inflation."""

    def __init__(self, pos_world_xy: np.ndarray, role: str = "unknown",
                 dt: float = 1.0) -> None:
        self._dt = dt
        self._H = np.zeros((2, 6))
        self._H[0, 0] = 1.0; self._H[1, 1] = 1.0  # observe x, y

        self._x = np.zeros(6)
        self._x[:2] = np.asarray(pos_world_xy, float)

        # High initial uncertainty; velocity and acceleration unknown
        self._P = np.diag([4.0, 4.0, 4.0, 4.0, 2.0, 2.0])

        params = ROLE_PARAMS[role]
        self._role = role
        self._F, self._Q = _singer_dynamics(params.alpha, params.sigma_a, dt)
        self._R = np.eye(2) * (params.measurement_noise ** 2)

    def predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, z_world_xy: np.ndarray) -> None:
        z = np.asarray(z_world_xy, float)
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.solve(S, np.eye(2))
        self._x = self._x + K @ (z - self._H @ self._x)
        I_KH = np.eye(6) - K @ self._H
        # Joseph form for numerical stability
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

    def mahalanobis_sq(self, z_world_xy: np.ndarray) -> float:
        """Squared Mahalanobis distance. Compare to _CHI2_95_2DOF for 95% gate."""
        z = np.asarray(z_world_xy, float)
        innovation = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        try:
            return float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return np.inf

    def switch_role(self, new_role: str) -> None:
        """Fix 2.1: swap motion model with P inflation to avoid overconfidence/divergence."""
        params = ROLE_PARAMS[new_role]
        new_F, new_Q = _singer_dynamics(params.alpha, params.sigma_a, self._dt)
        # Inflate P: add steady-state covariance of new model (Lyapunov solution)
        try:
            P_ss = solve_discrete_lyapunov(new_F, new_Q)
            self._P = self._P + 2.0 * P_ss
        except Exception:
            self._P = self._P * 4.0  # fallback: 4x inflation
        self._F, self._Q = new_F, new_Q
        self._R = np.eye(2) * (params.measurement_noise ** 2)
        self._role = new_role

    def cap_covariance(self, max_pos_var: float = 25.0) -> None:
        """Fix 2.3: prevent covariance blow-up during long Lost windows."""
        for i in range(2):
            if self._P[i, i] > max_pos_var:
                scale = max_pos_var / self._P[i, i]
                self._P = self._P * scale
                break

    def propagate_state(self, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_pred, P_pred) after n_frames of prediction, without mutating state."""
        x = self._x.copy()
        P = self._P.copy()
        for _ in range(n_frames):
            x = self._F @ x
            P = self._F @ P @ self._F.T + self._Q
        return x, P

    @property
    def pos_world_xy(self) -> np.ndarray:
        return self._x[:2].copy()

    @property
    def velocity_xy(self) -> np.ndarray:
        return self._x[2:4].copy()
