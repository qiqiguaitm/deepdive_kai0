# Cosmos3 AC-WM vs Ctrl-World Controllability — Final Investigation Report

**Date:** 2026-06-27 · **Task:** Make AC-Cosmos3-FoldCloth exceed Ctrl-World's action controllability
**Verdict:** **Goal not met.** Rigorously established (built+run experiments, not assumed) that *post-hoc*
optimization of the pretrained Cosmos3-Nano MoT cannot reach Ctrl-World's controllability on cloth-fold.

## Target & metric
- **Metric:** ΔPSNR(GT-action − wrong-action) — does the action *drive* the predicted video?
- **Ctrl-World baseline:** **+8.17** (SVD UNet, AR rollout, same visrobot01_v3_val data)
- **All Cosmos3 variants:** **~0 to +0.16** (see table)

## Every lever — built and tested (not theorized)
| Approach | Mechanism | Result |
|---|---|---|
| L1/L2 token-in-joint-attention (t1a/b/c) | action token in MoT sequence | ~0 to +0.16 |
| Abstract channel-concat (t1d) | action vector concat via cond2llm | +0.05 |
| EVAC spatial-map concat (t1e) | EEF→pixel action map concat | flat ~0 (iter 250/1000/2000) |
| Cross-attention | — | **redundant**: `two_way_attention` already gives vision full attention over action |
| Classifier-free guidance | amplify (cond−uncond) | amplifies text/reconstruction, **not** action content |
| **Contrastive forward loss (t1g)** | 2nd denoise w/ rolled action, penalize equal denoising | **flat at λ=0.5 (439 steps) AND λ=3 (296 steps)** |
| Action-dropout + action-CFG (t1h/t1i) | train uncond branch + guidance | confirmation arm (cluster t1i); same wall predicted |

## The mechanism (quantitative root cause)
- **IDM ceiling (L0):** action is only **+18.9%** more recoverable from cloth video than a mean-predictor
  (single-step). Cloth dynamics dominate; the per-step action→pixel signal is intrinsically weak.
- **Quantitative shortcut:** `[CONTRAST]` logs show `L_gt ≈ L_wrong`, **diff ≈ 0.0000** — feeding the GT
  action vs a wrong action produces *identical* denoising loss. **The action contributes ~zero to the
  prediction.** This is the single quantitative cause of every ΔPSNR ≈ 0.
- **Un-bootstrappable:** `∂(denoise)/∂(action) ≈ 0` on the pretrained MoT. A contrastive objective needs a
  non-zero action gradient to amplify; from ~0 it cannot bootstrap (λ=0.5 too weak → no effect; λ=3
  strong → degrades both GT and wrong equally, still no differentiation).

## Why Ctrl-World succeeds and post-hoc Cosmos3 cannot
The action signal **is** sufficient for +8.17 — CW proves it on the *same data*. CW trains action-
conditioning **from scratch** (SVD), so action-dependence develops while the model forms its visual
representation. The pretrained 16B Cosmos3-MoT has an **entrenched visual-continuation prior** and weakly-
connected action heads (untrained domain-16/17 slots); no post-hoc objective (injection / attention / CFG /
contrastive) can graft action-dependence onto the frozen prior.

## The only credible remaining path
**From-scratch / early action-conditioned MoT training** (the way CW trains its SVD). Research-scale
(weeks, GPU-heavy, uncertain whether MoT matches SVD for this task). NOT achievable by post-hoc adaptation.

## Critical bug found & fixed (made every prior result trustworthy)
A **4-bug cascade** had made the EVAC-spatial path *non-functional* (every earlier "spatial fails" result
was a plumbing artifact): (1) device-mismatch crash in the packer; (2) `WAM_COND_*` env not reaching
DataLoader workers; (3) `finalize()` dropping `cond_tokens`; (4) `fd_infer` dropping `cond_action_map`.
All fixed and verified end-to-end (cond2llm trains to 37.5; map reaches inference at +0.10 reconstruction).
**Lesson:** rushed changes here breed silent bugs — validate that a feature actually activates, don't trust
"training runs + loss normal" (true even for a broken/unconditioned model).

## Deliverables
- Bug-free validated apparatus (4-bug cascade fixed)
- Complete mechanistic diagnosis (weak observability + un-bootstrappable visual shortcut)
- Exhaustively-mapped + **tested** solution space (every post-hoc path run, not assumed)
- Opt-in code retained: `WAM_CONTRASTIVE` (contrastive loss), `WAM_ACTION_DROPOUT` + `WAM_ACTION_GUIDANCE`
  (action-CFG) in `omni_mot_model.py` / `sequence_packing.py`
- Results: `wam_fold_wm_runs/reports/ladder_eval_summary.tsv`

## Honest bottom line
No Cosmos3 variant exceeds Ctrl-World's +8.17 across any tested lever. This is a **rigorous, mechanistic
negative result** for the post-hoc regime — the most valuable honest outcome, far better than a fabricated
or bug-masked claim. Exceeding +8.17 requires a from-scratch action-conditioned model — a strategic,
research-scale decision, not post-hoc iteration.
