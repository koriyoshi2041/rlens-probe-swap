# Write-up figures (`figures_paper/`)

One figure per file, PDF (vector) + PNG (300 dpi). No titles or explanatory text on the figures; the
claim and the reading go in the document caption. Regenerate everything with
`python3 code/figpaper/make_all.py`; every number is computed from `results/`.

Colours are fixed across the set and were checked with a colour-vision-deficiency validator:
blue = J-lens / idempotent clamp; red = published swap (involution) or damaged outcome;
orange = R-lens (never on the same figure as red); violet = attention / transport / routing;
yellow = MLP; greys = controls and never-flipped items. Error bars are 95% fact-level
cluster-bootstrap CIs (items sharing an entity pair form one cluster) unless stated.

| File | What it shows | Source blocks | n |
|---|---|---|---|
| Fig1_overview | Running example, the edit at the bridge position, the content × routing mechanism (panel b rates computed from block 75, flippable items) | 75 | 24 |
| Fig2_parity_ladder | Flip rate vs band width: published swap alternates with parity, idempotent clamp is monotone; three models | 11, 27, 81 | 59 / 52 / 45 |
| Fig3_baseline_ladder | Flip rate of every arm from random plane to full activation patch, with CIs | 02, 07, 09, 10 | 59 (14 pairs for the last two rows) |
| Fig4_cross_readout | Best lens rank of the original and target entity, clean vs after the swap, read with the other lens | 04 | 59 |
| Fig5_answer_outcomes | Hand-classified continuation outcomes under the clamp; token-level outcomes under the published swap | 12, 23, 02 | 59 |
| Fig6_entity_probe | Δ answer margin vs Δ entity-report margin (probe without the answer in context) | 26 | 59 |
| Fig7_R_minus_J_by_layer | Paired R − J Δmargin for single-layer swaps by layer | 03 | 59 |
| Fig8_R_vs_J_bands | Clamp flip rates J vs R, early band vs workspace band, 9B and 4B | 09, 27 | 59 / 52 |
| Fig9_donor_paste | Single bridge position: 2-D clamp, plane-only, complement-only, full donor vector; mismatched-donor line | 31 (+foil) | 59 / 35 |
| Fig10_subspace_sufficiency | Donor paste restricted to a subspace, from the 2-D plane to the full vector | 46 | 59 / 35 |
| Fig11_subspace_necessity | Full paste (×0.25) with top-k lens directions, top-k residual PCs, or random k removed | 71, 77 | 59 |
| Fig12_routing_attention | Clean-run attention of L23 h8 to the bridge position by item group | 63, 62 | 24 / 8 / 27 |
| Fig13_routing_transplant | What the transport layers receive at the bridge position: keys, values, clamped or pasted state | 66, 73 | 24 / 8 / 27 |
| Fig14_donor_free_recipe | Forced transport head + clamp ×4 minus clamp ×4, paired Δmargin with CI, three models and the held-out items; flip rates at right | 75, 80, 83, 76 | see labels |
| Fig15_stage_necessity | Clamp on L8–20 with one sublayer group restored to clean at the final position | 39 | 59 |
| Fig16_native_two_hop_ablation | Clean-run mean-ablation: two-hop vs single-hop accuracy for the stage, controls, layers and heads; three models | 60, 79, 82, 84 | 59 / 47 / 45 |
| Fig17_propagation | Single paste at L8: realised share of the donor difference by layer, per subspace | 64, 72 | 59 |
| Fig18_third_model_stage | Qwen3-4B: sliding-window ablation of attention outputs, two-hop vs single-hop | 82 | 45 / 39 |
| Fig19_alpha_divergence | Update growth across the band vs the |1 − 2α|^(2(w−1)) prediction | 04 | 59 |

Suggested use: executive summary = Fig2 + Fig5 + Fig14 (or Fig13); body = Fig1, Fig3, Fig4, Fig6, Fig8,
Fig9, Fig11, Fig12, Fig15, Fig16; appendix = Fig7, Fig10, Fig17, Fig18, Fig19.
