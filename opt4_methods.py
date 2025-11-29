
import os, math, time, csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)
torch.set_default_dtype(torch.float64)

# -------------------------- Data --------------------------
def load_dataset(test_ratio=0.2, seed=0):
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        data = load_breast_cancer()
        X = data['data'].astype(np.float64)
        y = data['target'].astype(np.int64)  # {0,1}
        # Convert to {-1, +1}
        y = 2*y - 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test), True
    except Exception as e:
        # Fallback: synthetic separable
        n, d = 1000, 30
        rng = np.random.default_rng(0)
        w_true = rng.normal(size=d)
        X = rng.normal(size=(n, d))
        logits = X @ w_true / np.sqrt(d)
        y = np.where(logits > 0, 1, -1)
        # split
        idx = rng.permutation(n)
        ntr = int((1-test_ratio)*n)
        tr, te = idx[:ntr], idx[ntr:]
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        # standardize
        mu, sd = X_train.mean(0), X_train.std(0)+1e-8
        X_train = (X_train - mu)/sd
        X_test = (X_test - mu)/sd
        return torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test), False

# Utilities
def accuracy(w, b, X, y):
    with torch.no_grad():
        logits = X @ w + b
        preds = torch.where(logits >= 0, torch.tensor(1., dtype=logits.dtype), torch.tensor(-1., dtype=logits.dtype))
        return (preds.flatten() == y.to(dtype=preds.dtype)).double().mean().item()

def logistic_loss(w, b, X, y, lam=1e-3):
    z = y * (X @ w + b)
    return torch.log1p(torch.exp(-z)).mean() + 0.5*lam*(w@w)

def svm_primal_obj_hinge(w, b, X, y, C=1.0):
    # soft-margin hinge: 0.5||w||^2 + C * mean max(0, 1 - y*(Xw+b))
    z = y * (X @ w + b)
    return 0.5*(w@w) + C * torch.clamp(1 - z, min=0).mean()

def svm_primal_obj_sqh(w, b, X, y, C=1.0):
    z = y * (X @ w + b)
    return 0.5*(w@w) + C * torch.clamp(1 - z, min=0).pow(2).mean()

def plot_curves(hist, title, out_png, out_pdf):
    plt.figure()
    for k, v in hist.items():
        xs = list(range(1, len(v)+1))
        plt.plot(xs, v, label=k)
    plt.xlabel("Iteration / Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

def write_csv(rows, header, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def grad_norm_of_objective(w, b, X, y, kind, lam=1e-3, C=1.0):
    """Compute ||∇_{w,b} obj|| on given split, robust even if called under torch.no_grad()."""
    # 强制开启梯度环境，避免外层 no_grad 影响
    with torch.enable_grad():
        # 复制当前参数为叶子张量，并开启梯度
        w2 = w.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        # 选择对应目标函数
        if kind == "logistic":
            obj = logistic_loss(w2, b2, X, y, lam=lam)
        elif kind == "hinge":
            obj = svm_primal_obj_hinge(w2, b2, X, y, C=C)
        elif kind == "sqhinge":
            obj = svm_primal_obj_sqh(w2, b2, X, y, C=C)
        else:
            raise ValueError("unknown kind")

        # 计算梯度并给出二范数
        gw, gb = torch.autograd.grad(obj, [w2, b2], retain_graph=False, create_graph=False)
        # 显式 sum，避免形状歧义
        norm = torch.sqrt(torch.dot(gw, gw) + gb.pow(2).sum())
        return norm.detach().item()


# -------------------------- 1) IPM: logistic + L2-ball --------------------------
def ipm_logistic_l2ball(X, y, R=5.0, lam=1e-3, t0=1.0, mu=10.0, newton_tol=1e-8, max_newton=50, outer_iters=6):
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    logs = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
            "grad_norm_train": [], "grad_norm_test": []}

    def barrier_obj(w, b, t):
        base = logistic_loss(w, b, Xtr, ytr, lam=lam)
        barrier = -torch.log(R*R - (w@w) + 1e-12)
        return t*base + barrier

    def cg_solve(A_dot, b, max_iter=100, tol=1e-6):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        for _ in range(max_iter):
            Ap = A_dot(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-18)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < tol:
                break
            p = r + (rsnew/rsold) * p
            rsold = rsnew
        return x

    t = t0
    for outer in range(outer_iters):
        for _ in range(max_newton):
            obj = barrier_obj(w, b, t)
            grad_w, grad_b = torch.autograd.grad(obj, [w, b], create_graph=True)
            gnorm_newton = torch.sqrt(torch.dot(grad_w, grad_w) + grad_b.pow(2)).item()
            if gnorm_newton < newton_tol:
                break

            def hvp(vec):
                vec_w = vec[:d]
                vec_b = vec[d:]
                hv_w, hv_b = torch.autograd.grad(
                    (grad_w, grad_b), [w, b],
                    grad_outputs=(vec_w, vec_b),
                    retain_graph=True
                )
                return torch.cat([hv_w, hv_b]).detach()

            g_concat = torch.cat([grad_w, grad_b.view(1)])
            p = cg_solve(hvp, -g_concat.detach(), max_iter=200, tol=1e-6)

            step = 1.0
            with torch.no_grad():
                while step > 1e-8:
                    w_new = (w + step*p[:d]).detach().requires_grad_(True)
                    b_new = (b + step*p[d:]).detach().requires_grad_(True)
                    if (w_new @ w_new).item() >= R*R - 1e-9:
                        step *= 0.5
                        continue
                    val_new = barrier_obj(w_new, b_new, t).item()
                    if val_new <= obj.item() - 1e-4*step*(gnorm_newton**2):
                        w, b = w_new, b_new
                        break
                    step *= 0.5

            with torch.no_grad():
                tr_loss = logistic_loss(w, b, Xtr, ytr, lam=lam).item()
                te_loss = logistic_loss(w, b, Xte, yte, lam=lam).item()
                tr_acc = accuracy(w, b, Xtr, ytr)
                te_acc = accuracy(w, b, Xte, yte)
                gn_tr = grad_norm_of_objective(w, b, Xtr, ytr, kind="logistic", lam=lam)
                gn_te = grad_norm_of_objective(w, b, Xte, yte, kind="logistic", lam=lam)
                logs["train_loss"].append(tr_loss)
                logs["test_loss"].append(te_loss)
                logs["train_acc"].append(tr_acc)
                logs["test_acc"].append(te_acc)
                logs["grad_norm_train"].append(gn_tr)
                logs["grad_norm_test"].append(gn_te)
        t *= mu
    return w.detach(), b.detach(), logs

# -------------------------- 2) Dual projected gradient (SVM dual) --------------------------
def dual_projected_gradient_svm(X, y, C=1.0, iters=200, lr=None):
    n, d = X.shape
    y = y.double()
    K = (X @ X.T) * torch.outer(y, y) / n  # scale by n for stability
    if lr is None:
        q = torch.randn(n, dtype=K.dtype)
        for _ in range(30):
            q = K @ q
            q = q / (torch.linalg.norm(q) + 1e-12)
        L_est = torch.dot(q, K @ q).item()
        lr = 1.0 / (L_est + 1e-8)

    alpha = torch.zeros(n, dtype=K.dtype)
    logs = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
            "grad_norm_train": [], "grad_norm_test": []}

    def project_alpha(a):
        a = a - y * (torch.dot(y, a) / (torch.dot(y, y) + 1e-18))
        return torch.clamp(a, 0.0, C)

    for k in range(iters):
        grad = 1.0 - K @ alpha
        alpha = project_alpha(alpha + lr * grad)
        w = (X.T @ (alpha * y)) / n
        with torch.no_grad():
            idx = torch.where((alpha > 1e-6) & (alpha < C-1e-6))[0]
            if len(idx) > 0:
                b = (y[idx] - (X[idx] @ w)).mean()
            else:
                b = torch.tensor(0.0, dtype=w.dtype)
            tr_loss = svm_primal_obj_hinge(w, b, Xtr, ytr, C=C).item()
            te_loss = svm_primal_obj_hinge(w, b, Xte, yte, C=C).item()
            tr_acc = accuracy(w, b, Xtr, ytr); te_acc = accuracy(w, b, Xte, yte)
            gn_tr = grad_norm_of_objective(w, b, Xtr, ytr, kind="hinge", C=C)
            gn_te = grad_norm_of_objective(w, b, Xte, yte, kind="hinge", C=C)
        logs["train_loss"].append(tr_loss)
        logs["test_loss"].append(te_loss)
        logs["train_acc"].append(tr_acc)
        logs["test_acc"].append(te_acc)
        logs["grad_norm_train"].append(gn_tr)
        logs["grad_norm_test"].append(gn_te)
    return w.detach(), b.detach(), logs

# -------------------------- 3) Coordinate Descent (SMO) --------------------------
def smo_coordinate_descent_svm(X, y, C=1.0, iters=50):
    n, d = X.shape
    y = y.double()
    alpha = torch.zeros(n, dtype=X.dtype)
    b = 0.0
    logs = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
            "grad_norm_train": [], "grad_norm_test": []}
    K = (X @ X.T) / n

    def compute_w(alpha):
        return (X.T @ (alpha * y))

    for epoch in range(iters):
        for i in range(n):
            j = (i + epoch + 1) % n
            if j == i:
                j = (j + 1) % n
            Kii = K[i,i]; Kjj = K[j,j]; Kij = K[i,j]
            eta = Kii + Kjj - 2*Kij + 1e-12
            w = compute_w(alpha)
            f_i = (X[i] @ w).item() - b
            f_j = (X[j] @ w).item() - b
            E_i = f_i - y[i].item()
            E_j = f_j - y[j].item()
            s = y[i]*y[j]
            L, H = 0.0, C
            if y[i] != y[j]:
                L = max(0.0, (alpha[j]-alpha[i]).item())
                H = min(C, (C + alpha[j] - alpha[i]).item())
            else:
                L = max(0.0, (alpha[i] + alpha[j] - C).item())
                H = min(C, (alpha[i] + alpha[j]).item())
            if abs(eta) < 1e-12 or L==H:
                continue
            a_j = alpha[j] + y[j]*(E_i - E_j)/eta
            a_j = torch.tensor(min(max(a_j.item(), L), H), dtype=alpha.dtype)
            a_i = alpha[i] + s*(alpha[j] - a_j)
            alpha[i], alpha[j] = a_i, a_j
            w = compute_w(alpha)
            b = ((y[i] - X[i] @ w) + (y[j] - X[j] @ w))/2.0

        w = compute_w(alpha)
        tr_loss = svm_primal_obj_hinge(w, b, Xtr, ytr, C=C).item()
        te_loss = svm_primal_obj_hinge(w, b, Xte, yte, C=C).item()
        tr_acc = accuracy(w, b, Xtr, ytr); te_acc = accuracy(w, b, Xte, yte)
        gn_tr = grad_norm_of_objective(w, b, Xtr, ytr, kind="hinge", C=C)
        gn_te = grad_norm_of_objective(w, b, Xte, yte, kind="hinge", C=C)
        logs["train_loss"].append(tr_loss); logs["test_loss"].append(te_loss)
        logs["train_acc"].append(tr_acc); logs["test_acc"].append(te_acc)
        logs["grad_norm_train"].append(gn_tr); logs["grad_norm_test"].append(gn_te)

    return w.detach(), torch.tensor(b).detach(), logs

# -------------------------- 4) ADMM (squared hinge SVM) --------------------------
def admm_svm_squared_hinge(X, y, C=1.0, rho=1.0, iters=100):
    n, d = X.shape
    y = y.double()
    X = X.double()

    w = torch.zeros(d, dtype=X.dtype)
    b = torch.zeros(1, dtype=X.dtype)
    z = torch.zeros(n, dtype=X.dtype)
    u = torch.zeros(n, dtype=X.dtype)

    ones = torch.ones(n, dtype=X.dtype)
    Xt = X.T

    logs = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
            "grad_norm_train": [], "grad_norm_test": []}

    XTX = Xt @ X
    XT1 = Xt @ ones
    n_scalar = float(n)

    for k in range(iters):
        a = z - u
        yXa = Xt @ (y * a)
        rhs_w = rho * yXa
        rhs_b = rho * torch.dot(y, a)
        A11 = torch.eye(d, dtype=X.dtype) + rho * XTX
        A12 = rho * XT1.view(d, 1)
        A21 = rho * XT1.view(1, d)
        A22 = rho * torch.tensor([[n_scalar]], dtype=X.dtype)
        A_top = torch.cat([A11, A12], dim=1)
        A_bot = torch.cat([A21, A22], dim=1)
        A = torch.cat([A_top, A_bot], dim=0)
        rhs = torch.cat([rhs_w, rhs_b.view(1)], dim=0)
        sol = torch.linalg.solve(A, rhs)
        w = sol[:d]
        b = sol[d:]

        a = y * (X @ w + b) + u
        z_candidate = (2*C + rho*a) / (rho + 2*C)
        z = torch.minimum(z_candidate, torch.ones_like(z_candidate))
        mask = (a > 1.0)
        z[mask] = torch.maximum(a[mask], torch.ones_like(a[mask]))

        u = u + y * (X @ w + b) - z

        tr_loss = svm_primal_obj_sqh(w, b, Xtr, ytr, C=C).item()
        te_loss = svm_primal_obj_sqh(w, b, Xte, yte, C=C).item()
        tr_acc = accuracy(w, b, Xtr, ytr); te_acc = accuracy(w, b, Xte, yte)
        gn_tr = grad_norm_of_objective(w, b, Xtr, ytr, kind="sqhinge", C=C)
        gn_te = grad_norm_of_objective(w, b, Xte, yte, kind="sqhinge", C=C)
        logs["train_loss"].append(tr_loss); logs["test_loss"].append(te_loss)
        logs["train_acc"].append(tr_acc); logs["test_acc"].append(te_acc)
        logs["grad_norm_train"].append(gn_tr); logs["grad_norm_test"].append(gn_te)

    return w.detach(), b.detach(), logs

# -------------------------- Main --------------------------
if __name__ == "__main__":
    Xtr, ytr, Xte, yte, real = load_dataset()
    print("Dataset:", "Breast Cancer (sklearn)" if real else "Synthetic")
    n, d = Xtr.shape
    print("n_train =", n, "d =", d)

    # 1) IPM: logistic + L2-ball
    R = 5.0
    w1, b1, logs1 = ipm_logistic_l2ball(Xtr, ytr, R=R, lam=1e-3, t0=1.0, mu=8.0, outer_iters=5)
    plot_curves({"train": logs1["train_loss"], "test": logs1["test_loss"]},
                "IPM logistic loss", os.path.join(OUTDIR,"ipm_loss.png"), os.path.join(OUTDIR,"ipm_loss.pdf"))

    plot_curves({"train": logs1["train_acc"], "test": logs1["test_acc"]},
            "IPM accuracy", os.path.join(OUTDIR,"ipm_acc.png"), os.path.join(OUTDIR,"ipm_acc.pdf"))

    plot_curves({"train": logs1["grad_norm_train"], "test": logs1["grad_norm_test"]},
                "IPM grad norm", os.path.join(OUTDIR,"ipm_gradnorm.png"), os.path.join(OUTDIR,"ipm_gradnorm.pdf"))
    rows = list(zip(logs1["train_loss"], logs1["test_loss"], logs1["train_acc"], logs1["test_acc"],
                    logs1["grad_norm_train"], logs1["grad_norm_test"]))
    write_csv(rows, ["train_loss","test_loss","train_acc","test_acc","grad_norm_train","grad_norm_test"], os.path.join(OUTDIR,"ipm_metrics.csv"))

    # 2) Dual projected gradient (SVM dual)
    w2, b2, logs2 = dual_projected_gradient_svm(Xtr, ytr, C=1.0, iters=200)
    plot_curves({"train": logs2["train_loss"], "test": logs2["test_loss"]},
                "Dual-PG hinge loss", os.path.join(OUTDIR,"dualpg_loss.png"), os.path.join(OUTDIR,"dualpg_loss.pdf"))

    plot_curves({"train": logs2["train_acc"], "test": logs2["test_acc"]},
            "Dual-PG accuracy", os.path.join(OUTDIR,"dualpg_acc.png"), os.path.join(OUTDIR,"dualpg_acc.pdf"))

    plot_curves({"train": logs2["grad_norm_train"], "test": logs2["grad_norm_test"]},
                "Dual-PG grad norm (primal)", os.path.join(OUTDIR,"dualpg_gradnorm.png"), os.path.join(OUTDIR,"dualpg_gradnorm.pdf"))
    rows = list(zip(logs2["train_loss"], logs2["test_loss"], logs2["train_acc"], logs2["test_acc"],
                    logs2["grad_norm_train"], logs2["grad_norm_test"]))
    write_csv(rows, ["train_loss","test_loss","train_acc","test_acc","grad_norm_train","grad_norm_test"], os.path.join(OUTDIR,"dualpg_metrics.csv"))

    # 3) Coordinate Descent (SMO)
    w3, b3, logs3 = smo_coordinate_descent_svm(Xtr, ytr, C=1.0, iters=200)
    plot_curves({"train": logs3["train_loss"], "test": logs3["test_loss"]},
                "SMO-CD hinge loss", os.path.join(OUTDIR,"smo_loss.png"), os.path.join(OUTDIR,"smo_loss.pdf"))

    plot_curves({"train": logs3["train_acc"], "test": logs3["test_acc"]},
            "SMO-CD accuracy", os.path.join(OUTDIR,"smo_acc.png"), os.path.join(OUTDIR,"smo_acc.pdf"))

    plot_curves({"train": logs3["grad_norm_train"], "test": logs3["grad_norm_test"]},
                "SMO-CD grad norm (primal)", os.path.join(OUTDIR,"smo_gradnorm.png"), os.path.join(OUTDIR,"smo_gradnorm.pdf"))
    rows = list(zip(logs3["train_loss"], logs3["test_loss"], logs3["train_acc"], logs3["test_acc"],
                    logs3["grad_norm_train"], logs3["grad_norm_test"]))
    write_csv(rows, ["train_loss","test_loss","train_acc","test_acc","grad_norm_train","grad_norm_test"], os.path.join(OUTDIR,"smo_metrics.csv"))

    # 4) ADMM (squared hinge SVM)
    w4, b4, logs4 = admm_svm_squared_hinge(Xtr, ytr, C=1.0, rho=1.0, iters=200)
    plot_curves({"train": logs4["train_loss"], "test": logs4["test_loss"]},
                "ADMM squared-hinge loss", os.path.join(OUTDIR,"admm_loss.png"), os.path.join(OUTDIR,"admm_loss.pdf"))

    plot_curves({"train": logs4["train_acc"], "test": logs4["test_acc"]},
            "ADMM accuracy", os.path.join(OUTDIR,"admm_acc.png"), os.path.join(OUTDIR,"admm_acc.pdf"))

    plot_curves({"train": logs4["grad_norm_train"], "test": logs4["grad_norm_test"]},
                "ADMM grad norm (primal)", os.path.join(OUTDIR,"admm_gradnorm.png"), os.path.join(OUTDIR,"admm_gradnorm.pdf"))
    rows = list(zip(logs4["train_loss"], logs4["test_loss"], logs4["train_acc"], logs4["test_acc"],
                    logs4["grad_norm_train"], logs4["grad_norm_test"]))
    write_csv(rows, ["train_loss","test_loss","train_acc","test_acc","grad_norm_train","grad_norm_test"], os.path.join(OUTDIR,"admm_metrics.csv"))

    print("Done. Outputs in:", OUTDIR)
