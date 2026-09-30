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


def library_term_names():
    # names only depend on the fixed term list inside build_library
    dummy = torch.zeros(1, 1, 2, 2)
    return build_library(dummy, dummy)[1]


def format_equation(lam, names, lhs="u_t", tol=1e-8):
    # render nonzero coefficients as e.g. "u_t = -0.34*u*u_x +0.012*u_xx"
    lam = lam.detach().squeeze(-1).cpu()
    terms = [f"{c:+.4g}*{n}" for c, n in zip(lam.tolist(), names) if abs(c) > tol]
    return f"{lhs} = " + " ".join(terms) if terms else f"{lhs} = 0"


def sparse_regression(Phi, b, threshold=0.05, max_iter=10, alpha=1e-5):
    # Column-normalized STRidge: scale each library column to unit L2 norm
    # before regression so the sparsity threshold is comparable across terms
    # with wildly different magnitudes (e.g. "1" vs "u*u_x").
    # On unit-L2 columns a coefficient is raw_coef * rms(col) * sqrt(N), so the
    # threshold is scaled by sqrt(N) to make the cut mean "per-sample RMS
    # contribution to b >= threshold", independent of library row count.
    if b.dim() == 1:
        b = b.unsqueeze(-1)

    N, M = Phi.shape
    device, dtype = Phi.device, Phi.dtype
    thr = threshold * N ** 0.5

    col_norms = Phi.norm(dim=0).clamp_min(1e-12)
    Phi = Phi / col_norms

    eye_M = torch.eye(M, device=device, dtype=dtype)
    A_full = Phi.T @ Phi + alpha * eye_M
    rhs_full = Phi.T @ b
    lam = torch.linalg.solve(A_full, rhs_full)

    active = torch.ones(M, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        # identify currently active coefficients
        new_active = active & (lam.squeeze(-1).abs() >= thr)
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

    # Map coefficients back to original (unnormalized) column scale
    lam = lam / col_norms.unsqueeze(-1)
    return lam


def sparse_regression_diff(Phi, b, active_mask, alpha=1e-5):
    # Differentiable refit on the fixed active set, using the same column
    # normalization scheme as sparse_regression so threshold/scale match.
    if b.dim() == 1:
        b = b.unsqueeze(-1)
    _, M = Phi.shape
    device, dtype = Phi.device, Phi.dtype

    if int(active_mask.sum()) == 0:
        return torch.zeros(M, 1, device=device, dtype=dtype)

    active = active_mask.to(device=device).detach()
    Phi_Q = Phi[:, active]
    col_norms = Phi_Q.norm(dim=0).clamp_min(1e-12)
    Phi_Q_n = Phi_Q / col_norms

    k = Phi_Q.shape[1]
    eye_k = torch.eye(k, device=device, dtype=dtype)
    A = Phi_Q_n.T @ Phi_Q_n + alpha * eye_k
    rhs = Phi_Q_n.T @ b
    lam_Q_n = torch.linalg.solve(A, rhs)
    lam_Q = lam_Q_n / col_norms.unsqueeze(-1)

    lam = torch.zeros(M, 1, device=device, dtype=dtype)
    idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    lam = lam.index_copy(0, idx, lam_Q)

    return lam


def stridge_from_gram(G, c, n_rows, threshold=0.05, max_iter=10, alpha=1e-5):
    # STRidge from precomputed sufficient statistics G = Phi^T Phi (M x M) and
    # c = Phi^T b (M x 1). Used by PITALoss.precompute_true_coefficients so we
    # can aggregate the regression across the whole training set in one pass
    # without holding every Phi row in memory. n_rows is the total row count
    # behind G, needed to give the threshold the same per-sample-RMS meaning
    # as in sparse_regression.
    M = G.shape[0]
    device, dtype = G.device, G.dtype
    thr = threshold * n_rows ** 0.5

    d = G.diagonal().clamp_min(1e-24).sqrt()
    inv_d = 1.0 / d
    G_n = G * inv_d.unsqueeze(0) * inv_d.unsqueeze(1)
    c_n = c * inv_d.unsqueeze(-1)

    eye_M = torch.eye(M, device=device, dtype=dtype)
    lam_n = torch.linalg.solve(G_n + alpha * eye_M, c_n)

    active = torch.ones(M, dtype=torch.bool, device=device)
    for _ in range(max_iter):
        new_active = active & (lam_n.squeeze(-1).abs() >= thr)
        if new_active.sum() == 0:
            return torch.zeros(M, 1, device=device, dtype=dtype)
        if bool((new_active == active).all()):
            break
        active = new_active

        G_Q = G_n[active][:, active]
        c_Q = c_n[active]
        k = G_Q.shape[0]
        eye_K = torch.eye(k, device=device, dtype=dtype)
        lam_Q_n = torch.linalg.solve(G_Q + alpha * eye_K, c_Q)

        lam_n = torch.zeros(M, 1, device=device, dtype=dtype)
        lam_n[active] = lam_Q_n

    lam = lam_n * inv_d.unsqueeze(-1)
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

    def build_features(self, seq):
        # shared between discovery.forward() and PITALoss precompute: produce
        # flattened library and time derivatives without running regression
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
        Phi_flat = Phi.permute(0, 1, 3, 4, 2).reshape(-1, nT)
        du_dt_flat = du_dt.reshape(-1, 1)
        dv_dt_flat = dv_dt.reshape(-1, 1)
        return Phi_flat, du_dt_flat, dv_dt_flat

    def forward(self, seq, is_ground_truth: bool):
        # takes in sequence of tensors, builds library and runs algorithm for discoverying PDE
        Phi_flat, du_dt_flat, dv_dt_flat = self.build_features(seq)

        # Full Sparse Regression, no_grad since active-set selection does not
        # need gradient and saves compute time
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
            # active set comes from the regression's zeros: inactive terms are
            # exactly 0 there. Re-thresholding raw coefficients here would use
            # a different (raw-scale) criterion than the normalized-space one
            # sparse_regression pruned with.
            active_u = (lam_u_full.squeeze(-1) != 0).clone()
            active_v = (lam_v_full.squeeze(-1) != 0).clone()

        if is_ground_truth:
            # GROUNd truth is fixed target, return discovered coefficients detached
            lam_u = lam_u_full
            lam_v = lam_v_full
        else:
            # Predicted needs to be differentiable, resolved with grad enabled
            lam_u = sparse_regression_diff(
                Phi_flat,
                du_dt_flat,
                active_u,
                alpha=self.alpha,
            )
            lam_v = sparse_regression_diff(
                Phi_flat,
                dv_dt_flat,
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

        # Buffers populated by precompute_true_coefficients(); the consistency
        # loss target must be a single fixed Λ_true for the whole training set,
        # not a per-batch re-discovery of it (which would jitter every step).
        self.register_buffer("lam_u_true", torch.empty(0))
        self.register_buffer("lam_v_true", torch.empty(0))

    @torch.no_grad()
    def precompute_true_coefficients(self, dataloader, target_key="targets"):
        # Aggregate Phi^T Phi and Phi^T b across the entire training set, then
        # run STRidge once to get a fixed (Λ_u_true, Λ_v_true). Must be called
        # before the first forward() pass.
        device = self.log_sigma_data.device
        G = c_u = c_v = None
        M = None
        dtype = None
        n_rows = 0

        for batch in dataloader:
            true_seq = batch[target_key].to(device)
            Phi_flat, du_dt_flat, dv_dt_flat = self.discovery.build_features(true_seq)
            if G is None:
                M = Phi_flat.shape[1]
                dtype = Phi_flat.dtype
                G = torch.zeros(M, M, device=device, dtype=dtype)
                c_u = torch.zeros(M, 1, device=device, dtype=dtype)
                c_v = torch.zeros(M, 1, device=device, dtype=dtype)
            G = G + Phi_flat.T @ Phi_flat
            c_u = c_u + Phi_flat.T @ du_dt_flat
            c_v = c_v + Phi_flat.T @ dv_dt_flat
            n_rows += Phi_flat.shape[0]

        if G is None:
            raise RuntimeError("precompute_true_coefficients: dataloader was empty")

        lam_u = stridge_from_gram(
            G, c_u, n_rows,
            threshold=self.discovery.threshold,
            max_iter=self.discovery.max_iter,
            alpha=self.discovery.alpha,
        )
        lam_v = stridge_from_gram(
            G, c_v, n_rows,
            threshold=self.discovery.threshold,
            max_iter=self.discovery.max_iter,
            alpha=self.discovery.alpha,
        )
        self.lam_u_true = lam_u
        self.lam_v_true = lam_v

    @torch.no_grad()
    def print_discovered_equations(self, pred_seq=None):
        # human-readable view of the physics grounding: the fixed ground-truth
        # PDE from precompute_true_coefficients and, if a predicted rollout
        # [B, T, C, H, W] is passed, the PDE discovered from it
        names = library_term_names()
        if self.lam_u_true.numel():
            print("  ground truth: " + format_equation(self.lam_u_true, names, "u_t"))
            print("                " + format_equation(self.lam_v_true, names, "v_t"))
        else:
            print("  ground truth: <call precompute_true_coefficients first>")
        if pred_seq is not None:
            disc = self.discovery(pred_seq, is_ground_truth=True)
            print("  predicted:    " + format_equation(disc["lam_u"], names, "u_t"))
            print("                " + format_equation(disc["lam_v"], names, "v_t"))

    def forward(self, pred_seq, true_seq):
        if self.lam_u_true.numel() == 0:
            raise RuntimeError(
                "PITALoss.lam_u_true is empty; call precompute_true_coefficients(train_loader) once before training."
            )

        # ==== Data loss ====
        # relative L2 between predicted and true rollouts
        L_data = relative_l2(pred_seq, true_seq)

        # Predicted-trajectory PDE discovery (must be differentiable wrt model)
        pred_disc = self.discovery(pred_seq, is_ground_truth=False)

        Phi = pred_disc["Phi_flat"]
        du_dt = pred_disc["dudt_flat"]
        dv_dt = pred_disc["dvdt_flat"]
        lam_u_pred = pred_disc["lam_u"]
        lam_v_pred = pred_disc["lam_v"]

        # ==== Physics loss ====
        # mean-reduced so it lives on O(1) scale alongside L_data; sum-reduced
        # was 1e3 and drowned out the data and consistency terms
        res_u = ((Phi @ lam_u_pred) - du_dt).pow(2).mean()
        res_v = ((Phi @ lam_v_pred) - dv_dt).pow(2).mean()
        l0_u = pred_disc["active_u"].sum().to(Phi.dtype)
        l0_v = pred_disc["active_v"].sum().to(Phi.dtype)
        L_phy = res_u + res_v + self.alpha_l0 * (l0_u + l0_v)

        # ==== Consistency loss ====
        # pred coeffs vs precomputed fixed-target coeffs (mean reduction)
        L_con = ((self.lam_u_true - lam_u_pred) ** 2).mean() + ((self.lam_v_true - lam_v_pred) ** 2).mean()

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
