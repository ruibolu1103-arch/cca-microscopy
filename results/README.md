# Archived numerical results

These are the existing result files for **Align Before You Segment:
Label-Free Cross-Channel Alignment for Multi-Channel Fluorescence Microscopy**
by Ruibo Lu. They were copied without changing their numerical contents.
This packaging process did not rerun training or evaluation.

The numerical files here and `paper/numbers.json` are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribute Ruibo Lu,
the work above, and [the repository](https://github.com/ruibolu1103-arch/cca-microscopy).
Datasets and external models retain their provider terms.

| File | Contents |
|---|---|
| `final_tissuenet.json` | Main comparison, scene summaries, CCA seeds, bootstrap intervals, and alignment/harm tradeoffs. |
| `comir3seeds_tissuenet.json` | Three-seed CoMIR comparison with CCA. |
| `comir_tissuenet.json` | Original single-seed CoMIR comparison. |
| `comir_dead_tissuenet.json` | Post-hoc CoMIR dead-zone comparison. |
| `ablation_tissuenet.json` | CCA component and short-training ablations. |
| `cpsam_cca_tissuenet_test500.json` | Cellpose-SAM transfer on 500 test images. |
| `frame_test_tissuenet_test.json` | Test-set fixed-label versus moving-label coordinate-frame comparison. |
| `frame_test_tissuenet.json` | Validation-set coordinate-frame comparison. |
| `range_tissuenet.json` | Operating range under increasing displacement. |
| `fail_tissuenet.json` | Sparse and low-contrast image failure analysis. |
| `real_exemplar001.json` | MCMICRO exemplar-001 real cross-cycle results and marker summaries. |
| `final_deepliif.json` | DeepLIIF scene summaries and bootstrap results. |
| `bench.json` | Parameter counts and timing measurements. |
| `qual_tissuenet.json` | Qualitative sample selection metadata and numerical scores; no images. |
| `comir3seeds_tissuenet_rows.npz` | Per-image instance TP/FP/FN counts for CCA and CoMIR only. |

## Per-image coverage

For the 20-scene TissueNet alignment comparison, the released NPZ contains
per-image counts only for CCA and CoMIR, with three seeds of each. It does
not cover the end-to-end controls, MI, no alignment, or oracle alignment.
`final_tissuenet.json` and `final_deepliif.json` contain scene summaries and
bootstrap results, not complete per-image tables. `final_eval.py` constructs
per-image rows in memory without saving them. Other summary files must not
be described as complete per-image archives either.

NPZ keys are `scene|method`, for example `clean@0/s16|cca0`. Scene keys encode
degradation condition, moving stream, and displacement magnitude. Columns
are `inst_tp, inst_fp, inst_fn` (the definition in `reg_diag.py`). Rows follow
the test-image order used by `comir_eval.py`; no separate image-index array
was saved. Scene F1 pools TP/FP/FN counts before computing F1; it is not the
mean of individual image F1 scores.

Both `tissuenet/full/seed0/` and `deepliif/full/seed0/` also contain:

- `val_summary.json`: fixed segmenter validation metrics and selected thresholds.
- `val_per_image.json`: fixed segmenter validation image records.
- `test_summary.json`: fixed segmenter test summary metrics.
- `test_per_image.json`: fixed segmenter test image records.

Those JSON image records are segmenter diagnostics, not missing alignment
comparison rows. The archived threshold pairs are `[0.3, 0.5]`; several
evaluation scripts read the files directly. Recompute thresholds after
training a new fixed segmenter rather than assuming those values validate it.

## Qualitative export and figure generation

`paper/make_assets.py` reads the 14 top-level JSON files and, for qualitative
figures, `results/qual_tissuenet.npz`. The latter contains dataset images,
labels, predictions, and corrected images and is deliberately not distributed.
Generate it locally using `code/qual_export.py` after obtaining TissueNet
and training the necessary checkpoints; see the full commands in `README.md`.
The asset builder cannot finish all figures without that local file.

The selection rule in `qual_export.py` is descriptive: use the first `--n`
(default 300) test images of a seed-0 permutation; evaluate clean scenes
with nuclear stream shifted 16 pixels, membrane stream shifted 16 pixels,
and an aligned scene. For each displaced scene, among images with at least
30 cells choose samples nearest the 25th, 50th, and 75th percentiles of CCA
instance F1. If failures exist (shift error greater than 2 pixels), choose
the failure nearest the median cell count across both displaced scenes.
For the aligned scene, among images where gated MI applies a nonzero shift,
choose the image nearest the median MI harm. Methods are no alignment,
gated MI with tau 0.1, one-seed CCA with `tta_dead`, and oracle alignment
for displaced scenes. The code resolves percentile ties by first occurrence.

`reg_diag.py` and `reg_gate.py` retain historical validation diagnostics
needed as shared code dependencies; their diagnostic outputs are outside
this numerical archive. They use 600 pilot images after threshold/tuning
selection. These diagnostics are not substitutes for `final_eval.py`.
`selftest.py` is an optional implementation test and is not a result artifact.

The historical `--real` fine-tuning options of `train_cca.py` and
`train_comir.py` require the excluded pre-experiment module `real_data.py`.
The release rejects those options before loading models or data. The
published-method training path does not use them; `real_eval.py` remains
available for the reported real cross-cycle evaluation.
