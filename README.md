# FH-LoRA

This repository is the official anonymized implementation of *Frame-wise
Hyper-LoRA for Temporal Distillation from TimeSformer to ViT-Small*
(NeurIPS 2026 submission).

The paper studies temporal distillation from a frozen **TimeSformer**
teacher into a spatial **ViT-Small** student that has no temporal
attention. All temporal capacity is carried by frame-conditioned
**LoRA** adapters; the proposed module, **FH-LoRA** (Frame-wise
Hyper-LoRA), uses a single per-block hypernetwork trunk on the frame
index that feeds three projection-specific heads producing the
modulators $M_q(t), M_k(t), M_v(t)$.

The codebase is built on top of the official TimeSformer implementation
([`facebookresearch/TimeSformer`](https://github.com/facebookresearch/TimeSformer));
many design choices, configuration files, dataset loaders, and the
overall command-line surface mirror that repository.

## Requirements

The experiments use Python 3.9 and PyTorch 2.0 with CUDA 11.7. First,
create a conda environment and install PyTorch with the CUDA build that
matches your GPU driver:

```setup
conda create -n fh-lora python=3.9 -y
conda activate fh-lora
conda install pytorch=2.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then install the remaining packages:

```setup
pip install -r requirements.txt
```

### Teacher checkpoint

The TimeSformer teacher used for distillation is **not redistributed
here**. Please download the official `TimeSformer_divST_8_224_SSv2.pyth`
checkpoint (8-frame, 224-resolution, SSv2-finetuned) from the upstream
TimeSformer Model Zoo
(<https://github.com/facebookresearch/TimeSformer#model-zoo>) and place
the file under `checkpoints/`.

### Dataset: Something-Something V2

Please download the dataset and annotations from dataset provider.

Download the frame list from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

Extract the frames at 30 FPS. (We used ffmpeg-4.1.3 with command `ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"` in experiments.) Please put the frames in a structure consistent with the frame lists.

Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.

## Training

Training proceeds in two phases.

**Phase 1 — Spatial backbone distillation** distils the teacher's
spatial features into the ViT-Small student backbone (no adapters; all
student parameters trainable):

```train
torchrun --nproc_per_node=$N distill_spatial.py \
  --dataset ssv2 \
  --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --student_arch small \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/phase1 \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

**Phase 2 — Frame-conditioned LoRA distillation** freezes the Phase-1
backbone and trains the LoRA adapters (and, for hypernetwork variants,
the hypernetwork) against the same per-layer feature-matching loss. The
proposed configuration — **FH-LoRA**, $r{=}8$, trunk width $32$ —
corresponds to:

```train
torchrun --nproc_per_node=$N distill_temporal_lora.py \
  --dataset ssv2 \
  --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
  --student_weights output_ssv2/phase1/checkpoint_best.pth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --student_arch small \
  --lora_type fh_lora \
  --lora_rank 8 \
  --hyper_hidden_dim 32 \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/phase2_fh_lora \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

`--lora_type` selects the adapter family:

| `--lora_type`     | Paper name                  | Description                                                                   |
| ----------------- | --------------------------- | ----------------------------------------------------------------------------- |
| `standard`        | Standard LoRA               | Frame-blind LoRA on $Q,K,V$ (`M(t) = I`).                                     |
| `per_projection`  | Per-projection hypernetwork | One independent hypernetwork per projection.                                  |
| `fh_lora`         | FH-LoRA (proposed)          | One per-block trunk on the frame index, with three projection-specific heads. |

The wrapper script `run_ssv2_distill.sh` runs both phases end-to-end
with the proposed FH-LoRA configuration.

The teacher-free supervised baselines reported in Section 4.3 are
launched by `run_ssv2_supervised_baselines.sh`, which calls
`scripts/make_student_init_ckpt.py` to build a randomly-initialized
student (with or without FH-LoRA adapters) and then trains it directly
on the SSv2 labels via `eval_student.py --freeze_backbone False`.

The rank and trunk-width sweeps in Section 4.4 are reproduced by
`run_fhlora_rank_ablation.sh` (Figure 3a, Table 4),
`run_fhlora_hhd_ablation.sh` (Figure 3b, Table 5) and
`run_perproj_rank_ablation.sh` (Figure 3a comparison curve).

## Evaluation

Linear-probe evaluation on the SSv2 validation split, with a frozen
Phase-2 student:

```eval
torchrun --nproc_per_node=$N eval_student.py \
  --dataset ssv2 \
  --student_weights output_ssv2/phase2_fh_lora/checkpoint_best.pth \
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
  --output_dir output_ssv2/eval_fh_lora \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

Use `scripts/report_ssv2_eval_best.py` on `output_ssv2/eval_*/log.txt`
to tabulate best validation Top-1 / Top-5 across multiple runs.

To reproduce the teacher reference row of Table 1 (an upper bound
rather than a comparable probe — the teacher retains explicit temporal
attention and a fully fine-tuned classifier head):

```eval
python eval_teacher_ssv2.py \
  --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
  --data_path <ssv2_root>/annotations \
  --path_prefix <ssv2_root>/frames \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --output_dir output_ssv2/eval_teacher \
  --opts DATA.PATH_PREFIX <ssv2_root>/frames
```

## Pre-trained Models

* **TimeSformer teacher** (8-frame, 224-resolution, SSv2-finetuned):
  download `TimeSformer_divST_8_224_SSv2.pyth` from the upstream
  TimeSformer Model Zoo
  (<https://github.com/facebookresearch/TimeSformer#model-zoo>).
  Place the file under `checkpoints/`.

* **FH-LoRA student** (best configuration, $r{=}48$, $h_{\mathrm{hid}}{=}32$;
  29.70% Top-1 on SSv2): download
  `phase2_fh_lora_r48_hhd32.pth` from
  [`<download link to be inserted>`](TODO_LINK).
  Place the file under `checkpoints/` and pass it to `eval_student.py`
  via `--student_weights checkpoints/phase2_fh_lora_r48_hhd32.pth`
  (see the [Evaluation](#evaluation) section). The file is the full
  student state dict (ViT-Small backbone + FH-LoRA adapters + shared
  trunk + heads), so no separate Phase-1 download is required.

To reproduce the headline 29.70% number, run the evaluation command in
the next section with `--lora_rank 48 --hyper_hidden_dim 32` and the
checkpoint above.

## Results

Linear-probe accuracy on the Something-Something V2 validation split
(174 classes, best validation epoch). All adapter rows use a ViT-Small
student with no temporal attention in the trunk, 8 frames at 224×224,
adapter rank $r{=}8$, and trunk width $32$ where applicable. The
teacher row is an upper-bound reference rather than a comparable probe
since it retains the teacher's temporal attention layers and a fully
fine-tuned classifier head.

| Method                              | Adapter params | $r$ / trunk | Top-1 (%) | Top-5 (%) |
| ----------------------------------- | -------------- | ----------- | --------- | --------- |
| Teacher (reference)                 | —              | —           | 51.63     | 79.02     |
| Distilled backbone (no adapter)     | 0              | —           | 19.80     | 45.91     |
| + Standard LoRA  (`standard`)       | 221k           | 8 / —       | 21.05     | 47.56     |
| + Per-projection (`per_projection`) | 372k           | 8 / 32      | 28.15     | 55.26     |
| + FH-LoRA, proposed (`fh_lora`)     | 322k           | 8 / 32      | 27.99     | 54.55     |

The strongest configuration in the paper is FH-LoRA at $r{=}48$,
$h_{\mathrm{hid}}{=}32$, which reaches **29.70%** Top-1 on the same
linear-probe protocol (Table 4, Section 4.4). For the full rank
sweep, trunk-width sweep, and supervised teacher-free baselines, see
Tables 2–5 of the paper.

## Contributing

The majority of this work is licensed under **CC-BY-NC 4.0
International**, matching the upstream TimeSformer license. Portions of
the project are available under separate license terms: SlowFast and
`pytorch-image-models` (the basis of `models/vit_utils.py`,
`models/timesformer.py`, and the SSv2 dataset loader in `datasets/`)
are licensed under the **Apache 2.0** license.

This work is built on top of [TimeSformer](https://github.com/facebookresearch/TimeSformer),
[PySlowFast](https://github.com/facebookresearch/SlowFast), and
[`pytorch-image-models`](https://github.com/rwightman/pytorch-image-models).
If you use any component of this code, please cite these works as well:

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
