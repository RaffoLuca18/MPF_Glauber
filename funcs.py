####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the libraries                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import math
from tqdm import trange
import optax
from typing import Optional, Tuple, Dict



####################################################################################################
####################################################################################################
#                                                                                                  #
# some utilities                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def mask(J, t):
    """ masking without keeping the values """

    return (jnp.abs(J) >= t).astype(J.dtype)



####################################################################################################



def mask_prec(J, t):
    """ masking keeping the values """


    return J * (jnp.abs(J) >= t)



####################################################################################################
####################################################################################################
#                                                                                                  #
# simulating the evolution                                                                         #
#                                                                                                  #
####################################################################################################
####################################################################################################



def J_maker_nonsymmetric(n_spins, p = 0.2, minval = -1.0, maxval = 1.0, seed = 0):
    """ making the J matrix from a erdos-renyi graph structure """


    key = jax.random.PRNGKey(seed)
    precision = jnp.zeros((n_spins, n_spins))

    for i in range(n_spins):
        for j in range(n_spins):
            key, key_2, key_3 = jax.random.split(key, 3)
            u = jax.random.uniform(key_2)
            if u < p:
                rnd = jax.random.uniform(key_3, minval = minval, maxval = maxval)
                precision = precision.at[i, j].set(rnd)

    return precision



####################################################################################################



def glauber_parallel_jax(J, h, n_steps, sigma0 = None, return_trajectory = True, seed = 0):
    "simulating an ising model trajectory with parallel glauber dynamics (with external field h)"


    J = jnp.asarray(J, dtype=jnp.float32)
    n_spins = J.shape[0]
    key = jax.random.PRNGKey(seed)

    # external field
    if h is None:
        h_vec = jnp.zeros((n_spins,), dtype=jnp.float32)
    else:
        h_vec = jnp.asarray(h, dtype=jnp.float32).reshape((n_spins,))

    def init_sigma(k):
        u = jax.random.uniform(k, (n_spins,))
        return jnp.where(u < 0.5, -1, 1).astype(jnp.int32)

    if sigma0 is None:
        key, k0 = jax.random.split(key)
        sigma = init_sigma(k0)
    else:
        sigma = jnp.asarray(sigma0, dtype=jnp.int32).reshape((n_spins,))

    def one_step(carry, k_t):
        sigma_t = carry
        H_t = J @ sigma_t + h_vec                 # total field (interaction + external)
        p_plus = 1.0 / (1.0 + jnp.exp(-2.0 * H_t))
        u = jax.random.uniform(k_t, (n_spins,))
        sigma_tp1 = jnp.where(u < p_plus, 1, -1).astype(jnp.int32)
        return sigma_tp1, sigma_tp1

    keys = jax.random.split(key, n_steps)
    sigma_T, states = jax.lax.scan(one_step, sigma, keys)

    if return_trajectory:
        traj = jnp.concatenate([sigma[None, :], states], axis=0)
        return traj
    else:
        return sigma_T
    


####################################################################################################
####################################################################################################
#                                                                                                  #
# inference methods                                                                                #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _compute_lambda(alpha, n_spins, n_samples):
    """ compute the regularization strength lambda from the user-supplied coefficient alpha """


    return alpha * math.sqrt(math.log((n_spins ** 2) / 0.05) / n_samples)



####################################################################################################



def _reconstruct_single_spin_dnll(
    s,
    traj,                    # array (T, num_spins) with time-ordered samples
    method="D-NLL",
    lam=0.0,
    adj_row: Optional[jnp.ndarray] = None,
    n_steps=500,
    lr=1e-2,
    record_history=False,
):
    """
    Dynamic reconstruction (logistic D-NLL) for a single spin s.
    traj[t, i] = s_i^{(t)}  for t = 0..T-1
    """

    # build data (s^(t), s_i^(t+1))
    T, num_spins = traj.shape
    X = traj[:-1, :]          # s^{(t)}
    y = traj[1:, s]           # s_i^{(t+1)}
    num_conf = X.shape[0]
    n_samples = float(num_conf)

    # dynamic “nodal” statistics: y_t * s_j^{(t)}
    nodal_stat = (y[:, None] * X).at[:, s].set(y).astype(jnp.float32)

    # adjacency / L1 masks
    zero_mask = (
        (adj_row == 0) & (jnp.arange(num_spins) != s)
        if adj_row is not None
        else jnp.zeros(num_spins, dtype=bool)
    )

    # no L1 on the self-coupling, and no penalty on hard zeros
    base_l1 = jnp.ones(num_spins, dtype=jnp.float32).at[s].set(0.0)
    l1_mask = base_l1 * (~zero_mask).astype(jnp.float32)

    # trainable mask: 1 for parameters we optimize, 0 for fixed zeros
    train_mask = (~zero_mask).astype(jnp.float32)

    def loss_smooth(w_full_raw):
        # enforce hard zeros
        w_full = w_full_raw * train_mask

        # h_t = y_t * theta_i^{(t)}(J)
        h = nodal_stat @ w_full

        if method == "D-NLL":
            # time average: (1/T) * sum_t log(1 + exp(-2 h_t))
            return jnp.mean(jnp.log1p(jnp.exp(-2.0 * h)))
        else:
            # fallback: generic exponential loss on h
            return jnp.mean(jnp.exp(-1.0 * h))

    def objective(w_full_raw):
        w_full = w_full_raw * train_mask
        data_loss = loss_smooth(w_full)
        l1_term = lam * jnp.sum(l1_mask * jnp.abs(w_full))
        return data_loss + l1_term

    # initialize parameters
    params = jnp.zeros((num_spins,), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    history = []

    # optimization loop
    for t in range(1, n_steps + 1):
        val, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # re-enforce hard zeros
        params = params * train_mask

        if record_history:
            history.append(params)

    # final weights with hard zeros enforced
    w_full_final = params * train_mask
    hist_arr = jnp.stack(history, axis=0) if record_history else None

    return w_full_final, hist_arr



####################################################################################################



def _log_dynamics_likelihood(J, traj):
    """
    Dynamic log-likelihood  log P({s^{t+1}} | {s^t}, J)  for parallel Glauber dynamics.

    traj[t, i] = s_i^{(t)}   for t = 0..T-1
    """


    traj = jnp.asarray(traj)
    T, num_spins = traj.shape

    # build pairs (s^{(t)}, s^{(t+1)})
    X      = traj[:-1, :]    # s^{(t)}
    X_next = traj[1:, :]     # s^{(t+1)}
    n_trans = T - 1

    loglik_total = 0.0

    for s in range(num_spins):
        # y_t = s_i^{(t+1)}
        y = X_next[:, s]

        # nodal_stat[t, j] = y_t * s_j^{(t)}, with diagonal replaced by y_t
        nodal_stat = (y[:, None] * X).at[:, s].set(y).astype(jnp.float32)

        # h_t = y_t * θ_i^{(t)}(J)
        h = nodal_stat @ J[s, :]

        # log P(s_i^{(t+1)} | s^{(t)}) = h_t - log(2 cosh(h_t))
        # (cosh(h_t) = cosh(θ_i^{(t)}) because y_t = ±1)
        loglik_s = (h - jnp.log(2.0 * jnp.cosh(h))).sum()
        loglik_total = loglik_total + loglik_s

    return float(loglik_total)



####################################################################################################



def _oos_loss_dynamics(J, traj_out):
    """
    Out-of-sample dynamic negative log-likelihood,
    normalized per spin and per time step.
    """


    traj_out = jnp.asarray(traj_out)
    T, num_spins = traj_out.shape
    n_trans = T - 1  # number of transitions (t = 0..T-2)

    loglik = _log_dynamics_likelihood(J, traj_out)
    neg_loglik_per_spin_time = -loglik / (n_trans * num_spins)

    return float(neg_loglik_per_spin_time)



####################################################################################################



def inverse_glauber_trajectory(
    method,
    regularizing_value,
    symmetrization,
    traj,
    traj_out,
    adj = None,
    n_steps = 500,
    eps = 0.1,
    lr = 1e-2,
    record_history = True
    ):
    """ inverse ising """

    method = method.strip()
    symmetrization = symmetrization.strip().upper()

    traj = jnp.asarray(traj)
    n_samples, num_spins = traj.shape
    num_samples_float = float(n_samples)

    lam = _compute_lambda(regularizing_value, num_spins, num_samples_float)
    print(f"λ = {lam:.5g}  (reg = {regularizing_value})")

    W_snapshots: Dict[int, np.ndarray] = {}
    freq = jnp.ones((n_samples,), dtype=jnp.int32)/n_samples
    configs = traj

    spins = jnp.arange(num_spins)


    if adj is None:
        if record_history:
            def _worker(s):
                return _reconstruct_single_spin_dnll(
                    s, configs, method, lam, None,
                    n_steps=n_steps, lr=lr,
                    record_history=True,
                )
            rows, H = jax.vmap(_worker)(spins)
        else:
            def _worker(s):
                w, _ = _reconstruct_single_spin_dnll(
                    s, configs, method, lam, None,
                    n_steps=n_steps, lr=lr,
                    record_history=False,
                )
                return w
            rows = jax.vmap(_worker)(spins)
            H = None
    else:
        adj_rows = jnp.asarray(adj)
        if record_history:
            def _worker(s, adj_row):
                return _reconstruct_single_spin_dnll(
                    s, configs, method, lam, adj_row,
                    n_steps=n_steps, lr=lr,
                    record_history=True,
                )
            rows, H = jax.vmap(_worker, in_axes=(0,0))(spins, adj_rows)
        else:
            def _worker(s, adj_row):
                w, _ = _reconstruct_single_spin_dnll(
                    s, configs, method, lam, adj_row,
                    n_steps=n_steps, lr=lr,
                    record_history=False,
                )
                return w
            rows = jax.vmap(_worker, in_axes=(0,0))(spins, adj_rows)
            H = None

    W = rows

    if record_history and H is not None:
        H_np = np.asarray(H)
        _, T, _ = H_np.shape
        for step in range(T):
            W_snapshots[step] = H_np[:, step, :].astype(np.float32)

    if symmetrization == "Y":
        W = 0.5 * (W + W.T)
        if record_history:
            for k in list(W_snapshots.keys()):
                W_snapshots[k] = 0.5 * (W_snapshots[k] + W_snapshots[k].T)

    if traj_out is not None:
        print("Selecting best snapshot by out-of-sample log-likelihood...")

        best_W = W
        best_loss = _oos_loss_dynamics(best_W, traj_out)
        best_step = None

        if record_history and len(W_snapshots) > 0:
            for k, W_k_np in W_snapshots.items():
                W_k = jnp.asarray(W_k_np, dtype=jnp.float32)
                loss_k = _oos_loss_dynamics(W_k, traj_out)

                if loss_k < best_loss:
                    best_loss = loss_k
                    best_W = W_k
                    best_step = k

        if best_step is not None:
            print(f"  -> selected snapshot at step {best_step} ")
        else:
            print(f"  -> final weights already best with OOS -logL = {best_loss:.6g}")

        W = best_W

    W_np = np.asarray(W)
    history = {int(k): np.asarray(v) for k, v in W_snapshots.items()}

    return W_np, history



####################################################################################################
####################################################################################################
#                                                                                                  #
# bic procedure                                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def bic_procedure(traj_in, traj_out, J_hat, method, symmetrization, n_steps=200, early_stop=1.0):


    n_samples = len(traj_in)
    n_samples_out = len(traj_out)
    abs_val = jnp.sort(jnp.abs(J_hat).ravel())[::-1].tolist()
    k = math.ceil(early_stop * len(abs_val))
    abs_val = abs_val[:k]

    score = jnp.inf
    J_ret = J_hat
    bic = []
    maskeds = []

    for el in abs_val:
        mask_hat = mask_prec(J_hat, el)
        maskeds.append(mask_hat)
        score_try = (-2) * n_samples_out * _log_dynamics_likelihood(traj_out, mask_hat) + jnp.log(n_samples_out) * jnp.sum(mask(mask_hat, el))
        bic.append(score_try)
        if score_try < score:
            score = score_try
            J_ret = mask_hat

    # J_refit, _ = J_inference.inverse_ising(method, 0.0, symmetrization, samples, samples_out, adj=mask(J_ret, 1e-10), n_steps=n_steps)

    return J_ret, bic, score, maskeds
