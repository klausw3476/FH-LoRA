# FH-LoRA

This bundle contains anonymized supplementary code for a NeurIPS 2026
submission on **temporal distillation** from a frozen **TimeSformer**
teacher into a spatial **ViT-Small** student. All temporal capacity is
carried by frame-conditioned **LoRA** adapters; the proposed module,
**FH-LoRA** (Frame-wise Hyper-LoRA), uses a single per-block
hypernetwork trunk on the frame index that feeds three projection-specific
heads producing the modulators $M_q(t), M_k(t), M_v(t)$.

The codebase is built on top of the official TimeSformer implementation
([`facebookresearch/TimeSformer`](https://github.com/facebookresearch/TimeSformer));
many design choices, configuration files, dataset loaders, and the
overall command-line surface mirror that repository.

# Teacher checkpoint

The TimeSformer teacher used for distillation is **not redistributed
here**. Please download the official `TimeSformer_divST_8_224_SSv2.pyth`
checkpoint (8-frame, 224-resolution, SSv2-finetuned) from the upstream
TimeSformer Model Zoo:
<https://github.com/facebookresearch/TimeSformer#model-zoo>.

Place the downloaded `.pyth` file under `checkpoints/` and pass its path
through `--teacher_weights` (or set the `TEACHER` environment variable
read by the run scripts under `*.sh`; see the *Usage* section below).

Trained student weights (Phase-1 backbone and Phase-2 FH-LoRA adapters)
are not included with this submission; they will be released upon
acceptance.

# Installation

First, create a conda virtual environment and activate it:

```
conda create -n fh-lora python=3.9 -y
source activate fh-lora
```

Then install PyTorch with the CUDA build that matches your GPU driver
(the experiments use PyTorch 2.0 with CUDA 11.7):

* PyTorch: `conda install pytorch=2.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia`

Then install the remaining packages used by the codebase:

* `fvcore`: `pip install 'git+https://github.com/facebookresearch/fvcore'`
* `einops`: `pip install einops`
* `simplejson`: `pip install simplejson`
* `scikit-learn`: `pip install scikit-learn`
* `kornia`: `pip install kornia`
* OpenCV: `pip install opencv-python`
* PyAV: `conda install av -c conda-forge`
* `timm`: `pip install timm`

# Usage

## Dataset Preparation

### Something-Something V2

Please download the dataset and annotations from dataset provider.

Download the frame list from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

Extract the frames at 30 FPS. (We used ffmpeg-4.1.3 with command `ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"` in experiments.) Please put the frames in a structure consistent with the frame lists.

Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.

## Phase 1: Spatial Backbone Distillation

Phase 1 distils the teacher's spatial features into the ViT-Small
student backbone (no adapters; all student parameters trainable). With
8-frame, 224×224 SSv2 clips:

```
torchrun --nproc_per_node=$N distill_spatial.py \
  --dataset ssv2 \
  --teacher_weights /path/to/TimeSformer_divST_8_224_SSv2.pyth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --student_arch small \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/phase1 \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

The Phase-1 wrapper `run_ssv2_distill.sh` runs both this command and the
Phase-2 distillation below.

## Phase 2: Frame-Conditioned LoRA

Phase 2 freezes the Phase-1 backbone and trains the LoRA adapters (and,
for hypernetwork variants, the hypernetwork) against the same
per-layer feature-matching loss. The proposed configuration —
**FH-LoRA**, $r{=}8$, trunk width $32$ — corresponds to:

```
torchrun --nproc_per_node=$N distill_temporal_lora.py \
  --dataset ssv2 \
  --teacher_weights /path/to/TimeSformer_divST_8_224_SSv2.pyth \
  --student_weights output_ssv2/phase1/checkpoint_best.pth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --student_arch small \
  --lora_type fh_lora \
  --lora_rank 8 \
  --hyper_hidden_dim 32 \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/phase2_fhlora \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

`--lora_type` selects the adapter family:

| `--lora_type`     | Paper name                  | Description                                                                   |
| ----------------- | --------------------------- | ----------------------------------------------------------------------------- |
| `standard`        | Standard LoRA               | Frame-blind LoRA on $Q,K,V$ (`M(t) = I`).                                     |
| `per_projection`  | Per-projection hypernetwork | One independent hypernetwork per projection.                                  |
| `fh_lora`         | FH-LoRA (proposed)          | One per-block trunk on the frame index, with three projection-specific heads. |

The teacher-free supervised baselines reported in Section 4.3 are
launched by `run_ssv2_supervised_baselines.sh`, which calls
`scripts/make_student_init_ckpt.py` to build a randomly-initialized
student (with or without FH-LoRA adapters) and then trains it directly
on the SSv2 labels via `eval_student.py --freeze_backbone False`.

## Inference (linear probe)

Linear-probe evaluation on the SSv2 validation split, with a frozen
Phase-2 student, is launched by:

```
torchrun --nproc_per_node=$N eval_student.py \
  --dataset ssv2 \
  --student_weights output_ssv2/phase2_fhlora/checkpoint_best.pth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --num_labels 174 \
  --student_arch small \
  --use_tdlora True \
  --lora_type fh_lora \
  --lora_rank 8 \
  --hyper_hidden_dim 32 \
  --freeze_backbone True \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/eval_fhlora \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

Use `scripts/report_ssv2_eval_best.py` on `output_ssv2/eval_*/log.txt`
to tabulate best validation Top-1 / Top-5 across multiple runs.

## Reference evaluation of the teacher

To reproduce the teacher row of Table 1 (an upper-bound reference rather
than a comparable probe — the teacher retains explicit temporal
attention and a fully fine-tuned classifier head):

```
python eval_teacher_ssv2.py \
  --teacher_weights /path/to/TimeSformer_divST_8_224_SSv2.pyth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/eval_teacher \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

## Reproducing the rank and trunk-width sweeps

The sweeps in Section 4.4 (rank $r \in \{8,16,32,48,64,96\}$ and trunk
width $h_{\mathrm{hid}} \in \{8,16,32,48,64\}$) are reproduced by:

* `run_fhlora_rank_ablation.sh` — FH-LoRA rank sweep at
  $h_{\mathrm{hid}}=32$ (Figure 3a, Table 4).
* `run_fhlora_hhd_ablation.sh` — FH-LoRA trunk-width sweep at
  $r{\in}\{8, 48\}$ (Figure 3b, Table 5).
* `run_perproj_rank_ablation.sh` — per-projection rank sweep
  (Figure 3a, comparison curve).

Each script wraps the Phase-2 + linear-probe pipeline above.

# License

The majority of this work is licensed under **CC-BY-NC 4.0
International**, matching the upstream TimeSformer license. Portions of
the project are available under separate license terms: SlowFast and
`pytorch-image-models` (the basis of `models/vit_utils.py`,
`models/timesformer.py`, and the dataset loaders in `datasets/`) are
licensed under the **Apache 2.0** license.

# Acknowledgements

This work is built on top of [TimeSformer](https://github.com/facebookresearch/TimeSformer),
[PySlowFast](https://github.com/facebookresearch/SlowFast), and
[`pytorch-image-models`](https://github.com/rwightman/pytorch-image-models)
by Ross Wightman. We thank the authors for releasing their code. If you
use any component of this code, please cite these works as well:

```
@inproceedings{gberta_2021_ICML,
    author    = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title     = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    month     = {July},
    year      = {2021}
}

@misc{fan2020pyslowfast,
  author       = {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title        = {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year         = {2020}
}

@misc{rw2019timm,
  author       = {Ross Wightman},
  title        = {PyTorch Image Models},
  year         = {2019},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  doi          = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
