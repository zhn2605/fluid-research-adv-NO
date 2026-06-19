import torch
import torch.nn as nn
import torch.nn.functional as F

# Important to note pita is not an architecture but a wrapper with its own
# loss funciton


def time_derivative(u, dt=1.0):
    # central difference req. both neighors
    return (u[:, 2:] - u[:, :-2]) / (2.0 * dt)


def ddx(u, dx=1.0):
    # fill with zeros then calc derivative per region, central-diff with forward/ack diff at edges
    out = torch.zeros_like(u)
    out[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2.0 * dx)  # center region
    out[..., :, 0] = (u[..., :, 1] - u[..., :, 0]) / dx  # left col
    out[..., :, -1] = (u[..., :, -1] - u[..., :, -2]) / dx  # right col
    return out


def ddy(u, dy=1.0):
    out = torch.zeros_like(u)
    out[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dy)  # center region
    out[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dy  # top row
    out[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy  # bot row
    return out


def build_library(u, v, dx=1.0, dy=1.0):
    # 1st spatial derivatives
    u_x = ddx(u, dx)
    u_y = ddy(u, dy)
    v_x = ddx(v, dx)
    v_y = ddy(v, dy)

    # 2nd spatial deriatives
    u_xx = ddx(u_x, dx)
    u_yy = ddy(u_y, dy)
    v_xx = ddx(v_x, dx)
    v_yy = ddy(v_y, dy)

    ones = torch.ones_like(u)

    pairs = [
        ("1", ones),
        ("u", u),
        ("v", v),
        ("u^2", u * u),
        ("v^2", v * v),
        ("u*v", u * v),
        ("u_x", u_x),
        ("u_y", u_y),
        ("v_x", v_x),
        ("v_y", v_y),
        ("u_xx", u_xx),
        ("u_yy", u_yy),
        ("v_xx", v_xx),
        ("v_yy", v_yy),
        ("u*u_x", u * u_x),
        ("u*u_y", u * u_y),
        ("v*u_x", v * u_x),
        ("v*u_y", v * u_y),
        ("u*v_x", u * v_x),
        ("u*v_y", u * v_y),
        ("v*v_x", v * v_x),
        ("v*v_y", v * v_y),
    ]

    names = [n for n, _ in pairs]
    Phi = torch.stack([t for _, t in pairs], dim=2)

    return Phi, names


def sparse_regression(Phi, b, threshold=0.05, max_iter=10, alpha=1e-5):
    # b in this case represents target column, i.e. time derivative
    # alpha represents regularization strength for numerical stability
    # note this isnt the entire sparse_regression, but idrk what else to call this function
    if b.dim() == 1:
        b = b.unsqueeze(-1)
        N, M = Phi.shape
        device, dtype = Phi.device, Phi.dtype

        eye_M = torch.eye(M, device=device, dtype=dtype)
        A_full = Phi.T @ Phi + alpha * eye_M
        rhs_full = Phi.T @ b
        lam = torch.linalg.solve(A_full, rhs_full)

        active = torch.ones(M, dtype=torch.bool, device=device)

        for _ in range(max_iter):
            # identify currently active coefficients
            new_active = active & (lam.squeeze(-1).abs() >= threshold)
            if new_active.sum() == 0:
                # nothing passed threshold
                return torch.zeros(M, 1, device=device, dtype=dtype)
            if bool((new_active == active).all()):
                # active set didn't change anything
                break
            active = new_active

            # refit only using active columns / coefficients
            Phi_Q = Phi[:, active]
            k = Phi_Q.shape[1]
            eye_K = torch.eye(k, device=device, dtype=dtype)
            A_Q = Phi_Q.T @ Phi_Q + alpha * eye_K
            rhs_Q = Phi_Q.T @ b
            lam_Q = torch.linalg.solve(A_Q, rhs_Q)

            lam = torch.zeros(M, 1, device=device, dtype=dtype)
            lam[active] = lam_Q

            return lam


class PDEDiscovery(nn.Module):
    def __init__(self, dx=1.0, dy=1.0, dt=1.0, threshold=0.05, max_iter=10, alpha=1e-5, downsample=4):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.threshold = threshold  # beta
        self.max_iter = max_iter  # k

        assert downsample >= 1
        self.downsample = int(downsample)

    def forward(self, seq):
        # takes in sequence of tensors, builds library and runs algorithm for discoverying PDE
        s = self.downsample

        seq_ds = seq[..., ::s, ::s]  # stride downsample
        u_full = seq_ds[:, :, 0]
        v_full = seq_ds[:, :, 1]

        du_dt = (u_full[:, 2:] - u_full[:, :-2]) / (2.0 * self.dt)
        dv_dt = (v_full[:, 2:] - v_full[:, :-2]) / (2.0 * self.dt)

        u_mid = u_full[:, 1:-1]
        v_mid = v_full[:, 1:-1]
        Phi, _ = build_library(u_mid, v_mid, dx=self.dx * s, dy=self.dy * s)
        B, Tm, nT, h, w = Phi.shape

        # flatten Phi and values for sparse_regression compute compataility
        Phi_flat = Phi.permute(0, 1, 3, 4, 2)  # 22 features (lirary terms) are sorted last
        du_dt_flat = du_dt.reshape(-1, 1)
        dv_dt_flat = dv_dt.reshape(-1, 1)

        # Full Sparse Regression, no_grad since step does not block backprop and saves ocmpute time
        with torch.no_grad():
            lam_u = sparse_regression(
                Phi_flat.detach(),
                du_dt_flat.detach(),
                threshold=self.threshold,
                max_iter=self.max_iter,
                alpha=self.alpha,
            )
            lam_v = sparse_regression(
                Phi_flat.detach(),
                dv_dt_flat.detach(),
                threshold=self.threshold,
                max_iter=self.max_iter,
                alpha=self.alpha,
            )

        return {"Phi_flat": Phi_flat, "dudt_flat": du_dt_flat, "dvdt_flat": dv_dt_flat, "lam_u": lam_u, "lam_v": lam_v}


class PITALoss(nn.Module):
    def __init__(self, dx, dy, dt, threshold, max_iter, downsample):
        super().__init__()
        self.discovery = PDEDiscovery(
            dx=dx, dy=dy, dt=dt, threshold=threshold, max_iter=max_iter, downsample=downsample
        )
        self.L_data = nn.Parameter(torch.zeros(()))
        self.L_phy = nn.Parameter(torch.zeros(()))
        self.L_con = nn.Parameter(torch.zeros(()))

    def forward(self, pred_seq, true_seq):
        # computes difference between predicted PDE and true PDE
        L_data = F.mse_loss(pred_seq, true_seq)
