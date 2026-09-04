"""Toy check: first-order prediction of a sequential clamp.

Residual stream: 13 layers, each layer maps h -> h (identity) except that the lens
coordinate decays back toward its clean value by (1-rho) per layer (partial
re-derivation, q23 measured rho~0.95 one layer after the clamp). The margin is a
linear readout g.h at the end, so the true first-order prediction is exact here.

E3/q21 predict  sum_l <g_l, delta_l^clean>   (clean-coordinate deltas at every layer)
A consistent first-order prediction uses the deltas the clamp actually applies.
"""
import numpy as np
rng = np.random.default_rng(0)
d, L = 64, 13
v_s, v_t = rng.normal(size=d), rng.normal(size=d)
V = np.stack([v_s, v_t], 1)              # [d,2]
P = np.linalg.pinv(V)                    # [2,d]
g = rng.normal(size=d)                   # readout gradient (same at every layer: identity blocks)
h0 = rng.normal(size=d) * 3
for rho in (1.0, 0.95, 0.5, 0.0):
    # clean trajectory: identity blocks => same coords every layer
    c_clean = P @ h0
    exch = c_clean[[1, 0]]
    delta_clean = V @ (exch - c_clean)
    pred_clean_sum = L * (g @ delta_clean)
    # actual sequential clamp with re-derivation of coordinates between layers
    h = h0.copy(); pred_actual = 0.0
    for l in range(L):
        c = P @ h
        delta = V @ (exch - c)
        pred_actual += g @ delta
        h = h + delta
        # between layers: coordinates relax toward clean by (1-rho)
        c_now = P @ h
        h = h + V @ ((rho * c_now + (1 - rho) * c_clean) - c_now)
    true_effect = g @ (h - h0)
    print(f"rho={rho:4.2f}  E3-style prediction={pred_clean_sum:9.3f}  first-order w/ actual deltas={pred_actual:9.3f}  true effect={true_effect:9.3f}  E3 ratio={true_effect/pred_clean_sum:6.3f}")
