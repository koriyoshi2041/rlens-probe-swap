"""Pre-registered configuration for block 02 (main swap run). Fixed on 2026-09-02 19:55 CST.

Approved by Parafee at decision point 1 with four self-critique amendments:
separate Δlog p(answer) from Δlog p(swap_answer); shuffled-pair control as the main
random control; single-hop knowledge ceiling; last-5 position sensitivity added.
"""
BANDS = {"primary_L8_20": list(range(8, 21)), "early_L3_8": list(range(3, 9))}
PRIMARY_BAND = "primary_L8_20"
ALPHAS_MAIN = (0.5, 1.0, 2.0)
ALPHAS_EXPLORATORY = (4.0,)
ALPHAS_ALL = ALPHAS_MAIN + ALPHAS_EXPLORATORY
HEADLINE_ALPHA = 1.0
LENSES_SWAP = ("J", "R", "logit")
LENSES_CONTROL = ("J", "R")
CONTROL_ALPHAS = (1.0, 2.0)
SHUFFLE_SEEDS = (0, 1, 2)
GAUSS_SEEDS = (0,)
POSITION_SENSITIVITY = {"last3": [-3, -2, -1], "last5": [-5, -4, -3, -2, -1]}
SINGLE_LAYER_PROFILE = list(range(2, 25))
TOPK_SAVE = 20
QUAL_SEED = 0
QUAL_N = 8
GEN_TOKENS = 8
STOP_RULE = {"band": PRIMARY_BAND, "alpha": 2.0, "min_top1_swap_rate": 0.15, "min_median_delta_margin": 1.0}
