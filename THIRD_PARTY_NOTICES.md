# Third-party materials

The MIT license in `LICENSE` applies to the original project source code.
Numerical result files, including `paper/numbers.json`, are offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to
Ruibo Lu and the work identified in `README.md`. Neither license grants rights
to third-party datasets, external source code, or model weights.

No trained checkpoint, external model, raw dataset, processed image, or
dataset-derived qualitative array is distributed here. In particular,
trained task checkpoints are excluded because of third-party training-data
license restrictions. Obtain external materials from their providers and
follow their own terms.

| Component | Source and license context | Use and distribution |
|---|---|---|
| DINOv2 | [Official repository and license](https://github.com/facebookresearch/dinov2#license); [Base model card](https://huggingface.co/facebook/dinov2-base). Ordinary DINOv2 is identified as Apache-2.0 in the release source review. | Frozen ViT-B/14 features in `model.py` and `cca.py`. No upstream repository or backbone files are bundled. Download the ordinary Base model separately. |
| TissueNet v1.1 | [Download mirror](https://zenodo.org/records/15370205), [original data portal and terms](https://datasets.deepcell.org/), [dataset documentation](https://deepcell.readthedocs.io/en/latest/data-gallery/tissuenet.html). The original provider describes a modified Apache license for noncommercial academic use. A mirror's general license is insufficient to relicense this dataset. | Training and evaluation data, obtained by `download_data.py` and transformed by `prepare_data.py`. Images, labels, processed arrays, and trained checkpoints are excluded. |
| DeepLIIF | [Actual data record](https://zenodo.org/records/4751737). The record describes Apache-2.0 plus Commons Clause and noncommercial academic use for the upstream code; this does not establish unrestricted dataset or derived-model redistribution rights. | Training and evaluation data. This repository does not bundle upstream DeepLIIF code, data, or models. Consult the actual archive/provider terms. |
| CoMIR | [Official source](https://github.com/MIDA-group/CoMIR), [license](https://github.com/MIDA-group/CoMIR/blob/master/LICENSE). The upstream license is MIT with exceptions for some directories; inspect Tiramisu and dependency notices separately. | `comir.py` implements the project interface and integer-search registration. It imports `models.tiramisu.DenseUNet` only when supplied an external `--repo`. No external CoMIR source or weights are bundled. |
| Cellpose-SAM | [Official Cellpose source and license](https://github.com/MouseLand/cellpose), [model documentation](https://cellpose.readthedocs.io/). Verify terms for the exact source version and `cpsam_v2` model obtained. | Optional interfaces in `eval_cpsam.py` and `cpsam_cca.py`; install the package and obtain its model separately. Neither is bundled. |
| MCMICRO exemplar-001 | [Official example datasets](https://mcmicro.org/datasets/). Consult the sample's own data terms. | Real cross-cycle experiment in `real_eval.py`. Example TIFFs, illumination files, and markers are obtained separately. |

The available project records do not identify exact DINOv2 snapshot, CoMIR
commit, or Cellpose model revision. These are reproducibility limitations;
no revision identifiers have been invented. Record versions, commits, model
hashes, and provider terms when obtaining these materials.

Python dependencies are installed separately under their respective licenses.
`requirements.txt` records direct dependencies rather than an exact lockfile.
This document preserves the provenance identified in the local release manual;
it does not assert that external license pages were rechecked during packaging.
