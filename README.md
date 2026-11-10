# CCA for Multi-Channel Fluorescence Microscopy

Code and experimental results for:

**Align Before You Segment: Label-Free Cross-Channel Alignment for
Multi-Channel Fluorescence Microscopy**

Ruibo Lu  
School of Instrument Science and Technology, Xi'an Jiaotong University

- Repository: https://github.com/ruibolu1103-arch/cca-microscopy

## Method

Cross-Channel Alignment (CCA) is a label-free front end for correcting
translations between microscopy channels before segmentation. It combines
coarse correlation of projected frozen DINOv2 ViT-B/14 features with fine
correlation of lightweight convolutional features.

CCA is trained using synthetic translations and physics-based degradation
without cell annotations. The adopted inference variant, `tta_dead`, uses
the median of four rotation-based estimates, integer rounding, and a
one-pixel dead zone.

The evaluation reports recovery relative to oracle alignment together with
harm on already aligned inputs. Results on real cross-cycle images are
included as a limitation of the method's transfer across channel appearances.

## Repository layout

- `code/`: training, evaluation, data preparation, and analysis code.
- `paper/make_assets.py`: manuscript tables, figures, and numerical summaries.
- `paper/numbers.json`: archived numerical summaries.
- `results/`: archived numerical results and validation thresholds.

Raw datasets, processed images, trained checkpoints, external backbone files,
and manuscript sources or PDFs are not included.

Historical real-data fine-tuning options (`--real`) are outside this release.
The reported experiments do not use these options. Both training scripts reject
`--real` with a clear error before model or data loading.

## Environment

The recorded main environment used Python 3.13 and PyTorch 2.7.
The original experiments ran on Linux; timing results used an NVIDIA L40S.

The complete original package lockfile has not been recovered.
The following Windows PowerShell commands provide an installation starting
point using PyTorch 2.7.0 and CUDA 12.6. This patch version is an installation
example, not a verified record of the original environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install huggingface_hub transformers
```

Choose a PyTorch wheel compatible with the NVIDIA driver:
https://pytorch.org/get-started/previous-versions/

Cellpose-SAM was evaluated in a separate environment. Use the accompanying
Cellpose environment record when available; do not assume that installing
the latest Cellpose version reproduces the reported experiment.

Training and evaluation commands below require a CUDA GPU and can take
substantial time. Run them only when compute resources are available.

## Data

Obtain each dataset under its provider's terms. Dataset access does not
automatically grant permission to redistribute the images or trained weights.

The download script uses:

- TissueNet v1.1, distributed in the CellBinDB record:
  https://zenodo.org/records/15370205
- TissueNet original access and terms:
  https://datasets.deepcell.org/
- DeepLIIF:
  https://zenodo.org/records/4751737
- MCMICRO exemplar-001:
  https://mcmicro.org/datasets/

Download and prepare TissueNet and DeepLIIF from the repository root:

```powershell
python code/download_data.py data/raw
python code/prepare_data.py --raw data/raw --out data/proc
```

Expected archive MD5 checksums:

```text
42fd53473eb043e968815c03753d0ed1  tissuenet_v1.1.zip
f4812639cbee1f6732cb206e1e6acdc8  DeepLIIF_Training_Set.zip
3387c41b2d6771968976a1a6e67202dc  DeepLIIF_Validation_Set.zip
665899936c8012754524258ed16667ea  DeepLIIF_Testing_Set.zip
```

The experiments use the official dataset splits. TissueNet evaluation uses
all 1324 test images. DeepLIIF uses 575 training, 91 validation, and 598 test
images. The Cellpose-SAM transfer experiment uses a fixed 500-image test subset.

## External models and CoMIR

Download `config.json` and `model.safetensors` for `facebook/dinov2-base` into:

```text
weights/dinov2-base/
```

For example:

```powershell
hf download facebook/dinov2-base config.json model.safetensors --local-dir weights/dinov2-base
```

The original backbone snapshot revision was not recorded in the available
release materials; record the revision and file hashes you obtain. DINOv2 Small, Large, register variants, and Cell-DINO are not substitutes
for this backbone.

Obtain the CoMIR source separately:

```powershell
git clone https://github.com/MIDA-group/CoMIR.git external/CoMIR
```

The original upstream commit was not recorded in the available release
materials. Record your checkout commit and initialize any required dependencies. The supplied `--repo` directory must provide
`models.tiramisu.DenseUNet`; verify this before evaluation.

Our CoMIR baseline uses the authors' representation architecture and a
project-specific exhaustive integer search. Its timing should not be interpreted
as the timing of the original authors' registration pipeline.

For the Cellpose-SAM experiment, obtain the original `cpsam_v2` model from
the Cellpose provider and place it at:

```text
weights/cpsam/cpsam_v2
```

The original model revision and full Cellpose environment were not recovered;
record both when reproducing this optional experiment.

## Trained checkpoints

Trained checkpoints are not distributed with this repository because of
third-party training-data license restrictions. No checkpoint archive or
checkpoint download is provided. Obtain the datasets under their applicable
terms and run the **Train from scratch** commands below first. Here, training
from scratch means training the task models without the authors' checkpoints;
CCA and the segmenters still use the separately obtained frozen DINOv2 backbone.

Evaluation requires your own trained `last.pt` files at the paths below. Each
training command writes `last.pt` into its `--out` directory; `--run` and `--cca`
expect directories, whereas `--comir` expects the checkpoint file itself.

Important paths:

```text
results/cca/tissuenet/long0/last.pt
results/cca/tissuenet/long1/last.pt
results/cca/tissuenet/long2/last.pt

results/cca/deepliif/long0/last.pt
results/cca/deepliif/long1/last.pt
results/cca/deepliif/long2/last.pt

results/cca/tissuenet/seed0/last.pt

results/tissuenet/full/seed0/last.pt
results/tissuenet/full_mis/seed0/last.pt
results/tissuenet/full_xa/seed0/last.pt
results/tissuenet/unet_mis/seed0/last.pt

results/deepliif/full/seed0/last.pt
results/deepliif/full_xa/seed0/last.pt

results/comir/tissuenet/seed0/last.pt
results/comir/tissuenet/seed1/last.pt
results/comir/tissuenet/seed2/last.pt
```

`long0`, `long1`, and `long2` are independently trained CCA seeds 0, 1, and 2,
each trained for 18000 iterations. `cca/tissuenet/seed0` is the separate
6000-iteration ablation checkpoint.

The main fixed segmenter is `tissuenet/full/seed0`. CCA and segmenter checkpoints
exclude the frozen DINOv2 backbone, which must be obtained separately.

Archived validation threshold files are included with the numerical results
and are required by several evaluation scripts. After training your own fixed
segmenter, run `evaluate.py --split val` to produce thresholds for that model.
Do not treat the archived thresholds as validation of a new checkpoint.

## Train from scratch

Run from the repository root in PowerShell after preparing data and external
models. Define these paths before training or evaluation:

```powershell
$Data = 'data/proc'
$Snapshot = 'weights/dinov2-base'
$Results = 'results'
$CoMIR = 'external/CoMIR'
```

To train the fixed segmenter and TissueNet end-to-end controls:

```powershell
foreach ($Config in @('full', 'full_mis', 'full_xa', 'unet_mis')) {
    python code/train.py --root $Data --dataset tissuenet --config $Config --snapshot $Snapshot --out "$Results/tissuenet/$Config/seed0" --seed 0 --iters 12000 --batch 16 --workers 4
}
```

To generate the fixed segmenter's clean validation thresholds:

```powershell
python code/evaluate.py --root $Data --dataset tissuenet --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --out "$Results/tissuenet/full/seed0" --split val --workers 4
```

Train the CCA seeds and short-training ablation:

```powershell
foreach ($Seed in 0, 1, 2) {
    python code/train_cca.py --root $Data --dataset tissuenet --snapshot $Snapshot --out "$Results/cca/tissuenet/long$Seed" --seed $Seed --iters 18000 --batch 16 --workers 4
}

python code/train_cca.py --root $Data --dataset tissuenet --snapshot $Snapshot --out "$Results/cca/tissuenet/seed0" --seed 0 --iters 6000 --batch 16 --workers 4
```

Train CoMIR:

```powershell
foreach ($Seed in 0, 1, 2) {
    python code/train_comir.py --root $Data --dataset tissuenet --repo $CoMIR --out "$Results/comir/tissuenet/seed$Seed" --seed $Seed --iters 3000 --batch 24 --crop 128 --workers 4
}
```

Train the DeepLIIF fixed segmenter, control, and CCA seeds:

```powershell
foreach ($Config in @('full', 'full_xa')) {
    python code/train.py --root $Data --dataset deepliif --config $Config --snapshot $Snapshot --out "$Results/deepliif/$Config/seed0" --seed 0 --iters 12000 --batch 16 --workers 4
}
python code/evaluate.py --root $Data --dataset deepliif --snapshot $Snapshot --run "$Results/deepliif/full/seed0" --out "$Results/deepliif/full/seed0" --split val --workers 4
foreach ($Seed in 0, 1, 2) {
    python code/train_cca.py --root $Data --dataset deepliif --snapshot $Snapshot --out "$Results/cca/deepliif/long$Seed" --seed $Seed --iters 18000 --batch 16 --workers 4
}
```

Use a separate working copy when training or regenerating results to preserve
the archived numerical files. There are no archived checkpoints in this release.
Seeds do not guarantee bitwise-identical results across software and hardware.

## Reproduce the main comparison

Run the following from the repository root after preparing data, training the
required checkpoints, and validating your fixed segmenter. These commands
write results; use a working copy to preserve the archived numerical files:

```powershell
$Data = 'data/proc'
$Snapshot = 'weights/dinov2-base'
$Results = 'results'
$CoMIR = 'external/CoMIR'

python code/final_eval.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --controls "full_xa=$Results/tissuenet/full_xa/seed0" "full_mis=$Results/tissuenet/full_mis/seed0" "unet_mis=$Results/tissuenet/unet_mis/seed0" --cca "$Results/cca/tissuenet/long0" "$Results/cca/tissuenet/long1" "$Results/cca/tissuenet/long2" --variant tta_dead --workers 4 --out $Results

python code/comir_eval.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" "$Results/cca/tissuenet/long1" "$Results/cca/tissuenet/long2" --comir "$Results/comir/tissuenet/seed0/last.pt" "$Results/comir/tissuenet/seed1/last.pt" "$Results/comir/tissuenet/seed2/last.pt" --repo $CoMIR --workers 4 --out $Results
```

These produce `final_tissuenet.json`, `comir3seeds_tissuenet.json`, and the
three-seed CCA/CoMIR per-image count archive.

## Reproduce additional comparisons

Single-seed CoMIR and the post-hoc dead-zone check:

```powershell
python code/comir_eval.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" "$Results/cca/tissuenet/long1" "$Results/cca/tissuenet/long2" --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --workers 4 --out $Results

python code/comir_eval.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" "$Results/cca/tissuenet/long1" "$Results/cca/tissuenet/long2" --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --dead --workers 4 --out $Results
```

Component ablation and coordinate-frame experiments:

```powershell
python code/ablation_eval.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" --short "$Results/cca/tissuenet/seed0" --workers 4 --out $Results

python code/frame_test.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --split val --workers 4 --out $Results

python code/frame_test.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --split test --workers 4 --out $Results
```

Operating range and failure analysis:

```powershell
python code/range_eval.py --root $Data --snapshot $Snapshot --cca "$Results/cca/tissuenet/long0" --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --workers 4 --out $Results

python code/fail_eval.py --root $Data --snapshot $Snapshot --cca "$Results/cca/tissuenet/long0" --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --workers 4 --out $Results
```

DeepLIIF:

```powershell
python code/final_eval.py --root $Data --snapshot $Snapshot --dataset deepliif --run "$Results/deepliif/full/seed0" --controls "full_xa=$Results/deepliif/full_xa/seed0" --cca "$Results/cca/deepliif/long0" "$Results/cca/deepliif/long1" "$Results/cca/deepliif/long2" --variant tta_dead --workers 4 --out $Results
```

DeepLIIF results are descriptive because native channel or label offsets
confound the clean and synthetic-oracle references.

After downloading and extracting MCMICRO exemplar-001 so that the selected
directory contains `raw/`, `illumination/`, and `markers.csv`:

```powershell
python code/real_eval.py --data external/exemplar-001 --snapshot $Snapshot --cca "$Results/cca/tissuenet/long0" "$Results/cca/tissuenet/long1" "$Results/cca/tissuenet/long2" --variant tta_dead --workers 4 --out $Results
```

Run Cellpose-SAM in its separate, documented environment:

```powershell
python code/cpsam_cca.py --root $Data --snapshot $Snapshot --model weights/cpsam/cpsam_v2 --cca "$Results/cca/tissuenet/long0" --split test --limit 500 --variant tta_dead --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --out $Results
```

Benchmarking requires an idle GPU. Timings from a different GPU are not expected
to match the manuscript's L40S measurements:

```powershell
python code/bench.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" --comir "$Results/comir/tissuenet/seed0/last.pt" --repo $CoMIR --out $Results
```

## Rebuild manuscript assets

The archived numerical JSON files allow inspection of the reported tables
without rerunning model evaluation.

The complete asset builder also needs `results/qual_tissuenet.npz`, which
contains dataset-derived images and labels. It is not distributed. Generate it locally from lawfully obtained TissueNet
data after training the required checkpoints:

```powershell
python code/qual_export.py --root $Data --snapshot $Snapshot --run "$Results/tissuenet/full/seed0" --cca "$Results/cca/tissuenet/long0" --n 300 --workers 4 --out $Results
```

Then:

```powershell
$env:OMP_NUM_THREADS = '2'
python paper/make_assets.py
```

The asset builder writes tables, figures, and `paper/numbers.json`.
This release includes the asset builder and archived numbers only. The full
builder needs the locally generated qualitative array to finish all figures.

## Result formats and coverage

`results/comir3seeds_tissuenet_rows.npz` stores per-image arrays under keys of
the form `scene|method`. Columns are:

```text
inst_tp, inst_fp, inst_fn
```

Rows follow the test-image order. Scene names identify the degradation condition,
moving stream, and translation magnitude.

Scene-level instance F1 is computed from pooled counts, rather than by averaging
the F1 scores of individual images.

The archived `final_*.json` files contain scene summaries and bootstrap results.
They must not be described as complete per-image archives. See `results/README.md`
for the exact coverage of released per-image files.

## License and third-party materials

Original source code is licensed under MIT; see `LICENSE`.
This license does not relicense third-party datasets, backbone weights, or
components obtained from other repositories.

The released numerical result files, including `paper/numbers.json`, are
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Trained checkpoints are excluded because of third-party training-data license
restrictions. This repository grants no rights to excluded datasets or models.
Dataset-derived images and labels retain the applicable provider terms.
See `THIRD_PARTY_NOTICES.md` for provenance and notices.

## Citation

Please cite the work by its title and author and identify the repository
revision used in your work.

```bibtex
@misc{lu_cca_microscopy,
  author = {Lu, Ruibo},
  title = {Align Before You Segment: Label-Free Cross-Channel Alignment
           for Multi-Channel Fluorescence Microscopy},
  url = {https://github.com/ruibolu1103-arch/cca-microscopy}
}
```

Machine-readable software citation metadata are provided in `CITATION.cff`.
Please also cite TissueNet, DeepLIIF, DINOv2, CoMIR, Cellpose-SAM, and MCMICRO
when using their respective data or components.
