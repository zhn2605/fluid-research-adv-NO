import torch
import torch.nn as nn
import torch.nn.functional as F

# Important to note pita is not an architecture but a wrapper with its own
# loss funciton


def relative_l2(pred_seq, true_seq, eps=1e-8):
    B, T = pred_seq.shape[:2]
    pred_flat = pred_seq.reshape(B, T, -1)
    true_flat = true_seq.reshape(B, T, -1)
    diff_norm = torch.norm(pred_flat - true_flat, p=2, dim=-1)
    true_norm = torch.norm(true_flat, p=2, dim=-1).clamp_min(eps)
    rel = diff_norm / true_norm
    return rel.mean()


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

    _, M = Phi.shape
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


def sparse_regression_diff(Phi, b, active_mask, alpha=1e-5):
    # active_maskl = bool tensor of shape [N_terms] that holds library columns that passed sparse regersion
    if b.dim() == 1:
        b = b.unsqueeze(-1)
    _, M = Phi.shape
    device, dtype = Phi.device, Phi.dtype

    if int(active_mask.sum()) == 0:
        return torch.zeros(M, 1, device=device, dtype=dtype)

    active = active_mask.to(device=device).detach()
    Phi_Q = Phi[:, active]
    k = Phi_Q.shape[1]
    eye_k = torch.eye(k, device=device, dtype=dtype)
    A = Phi_Q.T @ Phi_Q + alpha * eye_k
    rhs = Phi_Q.T @ b
    lam_Q = torch.linalg.solve(A, rhs)

    lam = torch.zeros(M, 1, device=device, dtype=dtype)
    idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    lam = lam.index_copy(0, idx, lam_Q)

    return lam


class PDEDiscovery(nn.Module):
    def __init__(self, dx=1.0, dy=1.0, dt=1.0, threshold=0.05, max_iter=10, alpha=1e-5, downsample=4):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.threshold = threshold  # beta
        self.max_iter = max_iter  # k
        self.alpha = alpha

        assert downsample >= 1
        self.downsample = int(downsample)

    def forward(self, seq, is_ground_truth: bool):
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
        Phi_flat = Phi.permute(0, 1, 3, 4, 2).reshape(-1, nT)  # 22 features (lirary terms) are sorted last
        du_dt_flat = du_dt.reshape(-1, 1)
        dv_dt_flat = dv_dt.reshape(-1, 1)

        # Full Sparse Regression, no_grad since step does not block backprop and saves ocmpute time
        with torch.no_grad():
            lam_u_full = sparse_regression(
                Phi_flat.detach(),
                du_dt_flat.detach(),
                threshold=self.threshold,
                max_iter=self.max_iter,
                alpha=self.alpha,
            )
            lam_v_full = sparse_regression(
                Phi_flat.detach(),
                dv_dt_flat.detach(),
                threshold=self.threshold,
                max_iter=self.max_iter,
                alpha=self.alpha,
            )
            active_u = (lam_u_full.squeeze(-1).abs() >= self.threshold).clone()
            active_v = (lam_v_full.squeeze(-1).abs() >= self.threshold).clone()

        if is_ground_truth:
            # GROUNd truth is fixed target, return discovered coefficients detached
            lam_u = lam_u_full
            lam_v = lam_v_full
        else:
            # Predicted needs to be differentiable, resolved with grad enabled
            lam_u = sparse_regression_diff(
                Phi_flat.detach(),
                du_dt_flat.detach(),
                active_u,
                alpha=self.alpha,
            )
            lam_v = sparse_regression_diff(
                Phi_flat.detach(),
                dv_dt_flat.detach(),
                active_v,
                alpha=self.alpha,
            )

        return {
            "Phi_flat": Phi_flat,
            "dudt_flat": du_dt_flat,
            "dvdt_flat": dv_dt_flat,
            "lam_u": lam_u,
            "lam_v": lam_v,
            "active_u": active_u,
            "active_v": active_v,
        }


class PITALoss(nn.Module):
    def __init__(self, dx=1.0, dy=1.0, dt=1.0, threshold=0.05, max_iter=10, alpha=1e-5, alpha_l0=1e-4, downsample=4):
        super().__init__()
        self.discovery = PDEDiscovery(
            dx=dx, dy=dy, dt=dt, threshold=threshold, max_iter=max_iter, alpha=alpha, downsample=downsample
        )
        self.alpha_l0 = alpha_l0
        self.log_sigma_data = nn.Parameter(torch.zeros(()))
        self.log_sigma_phy = nn.Parameter(torch.zeros(()))
        self.log_sigma_con = nn.Parameter(torch.zeros(()))

    def forward(self, pred_seq, true_seq):
        # ==== Data loss ====
        # computes difference between predicted PDE and true PDE
        L_data = relative_l2(pred_seq, true_seq)

        # ground truth & pred PDE Discovery
        with torch.no_grad():
            true_disc = self.discovery(true_seq, is_ground_truth=True)
            lam_u_true = true_disc["lam_u"]
            lam_v_true = true_disc["lam_v"]

            pred_disc = self.discovery(pred_seq, is_ground_truth=False)

            Phi = pred_disc["Phi_flat"]
            du_dt = pred_disc["dudt_flat"]
            dv_dt = pred_disc["dvdt_flat"]
            lam_u_pred = pred_disc["lam_u"]
            lam_v_pred = pred_disc["lam_v"]

        # ==== Physics loss ====
        res_u = ((Phi @ lam_u_pred) - du_dt).pow(2).sum()
        res_v = ((Phi @ lam_v_pred) - dv_dt).pow(2).sum()
        # L0 norm = count of non-zero entries constant (active set size)
        l0_u = pred_disc["active_u"].sum().to(Phi.dtype)
        l0_v = pred_disc["active_v"].sum().to(Phi.dtype)
        L_phy = res_u + res_v + self.alpha_l0 * (l0_u + l0_v)

        # ==== Consistency loss ====
        L_con = ((lam_u_true - lam_u_pred) ** 2).sum() + ((lam_v_true - lam_v_pred) ** 2).sum()

        # === Unertainty-weighed total ===
        precision_data = 0.5 * torch.exp(-2.0 * self.log_sigma_data)
        precision_phy = 0.5 * torch.exp(-2.0 * self.log_sigma_phy)
        precision_con = 0.5 * torch.exp(-2.0 * self.log_sigma_con)

        total = (
            precision_data * L_data
            + precision_phy * L_phy
            + precision_con * L_con
            + self.log_sigma_data
            + self.log_sigma_phy
            + self.log_sigma_con
        )

        parts = {
            "L_data": L_data.detach(),
            "L_phy": L_phy.detach(),
            "L_con": L_con.detach(),
        }

        return total, parts
