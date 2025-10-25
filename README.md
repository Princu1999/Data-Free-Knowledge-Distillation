# Adversarial Knowledge Distillation (AKD) with GANs — CIFAR-100

This repository provides a production-ready, modular PyTorch implementation of **data-free adversarial knowledge distillation**: a pretrained **ResNet-34 teacher** supervises compact **ResNet-18 student** variants while a lightweight **generator** synthesizes training stimuli on the fly. The codebase separates concerns into clear modules (models, engine, utils, scripts), supports reproducible runs, and mirrors the original training dynamics (alternating student–generator updates) for faithful results.

**Why this repo**

- **Data-free KD**: distill without access to original training data via generator-driven synthesis.  
- **Compact students**: 50% and ~20% parameter variants for efficient deployment.  
- **Clean structure**: configuration, training, evaluation, and logging are neatly modularized.  
- **Drop-in usage**: one-command training/evaluation with optional teacher checkpoint loading.

---

## ✨ Highlights
- **Teacher**: ResNet-34 (pretrained on CIFAR-100) distills logits to students without real training data.  
- **Students**: (i) `ResNet18_8x_Small` (~20% teacher params), (ii) `ResNet18_8x` (~50% teacher params).  
- **GeneratorA**: Lightweight GAN-style image sampler (latent `z→32×32`) to drive KD.

## 🚀 Quickstart
```bash
pip install -r requirements.txt
# Optional: add teacher weights at ./teacher/best_resnet34_cifar100.pth
python scripts/train_akd.py
# Sample synthetic images
pytho

To sample images from the generator:
```bash
python scripts/generate_images.py
```

## 🗂 Project Structure
```
akd_kd_gan/
  __init__.py
  config.py                 # AKDConfig: paths, hparams, seeds, run folders
  data.py                   # CIFAR‑100 loader for test‑only split
  losses.py                 # KL‑KD, diversity loss (utility)
  akd_kd_gan/
    models/
      generator.py            # GeneratorA (z→32×32)
      resnet_small.py         # BasicBlock/Bottleneck/ResNet/ResNet18_8x/_Small
    engine/
      train.py                # train_epoch() (alternate student/gen updates)
      eval.py                 # evaluate() → CE loss & accuracy
    utils/
      logger.py, early_stopping.py, checkpoint.py
  scripts/
    train_akd.py              # end‑to‑end AKD training loop (teacher→students)
    generate_images.py        # grid + individual PNGs
requirements.txt
README.md
```

## Results

**Summary**
- Compact students achieve competitive CIFAR‑100 top‑1 accuracy while reducing parameter count by up to **~8×** vs. the teacher.
- All evaluations are on **test-only** splits (original training data is not used). The generator synthesizes 32×32 inputs for AKD.

### Top‑1 Accuracy by Test Split (%)
| Split (test subset) | Student‑50 (ResNet18_8x) | Student‑20 (ResNet18_8x_Small) |
|---:|:---:|:---:|
| 20% of CIFAR‑100 test | **39.90** | **20.00** |
| 10% of CIFAR‑100 test | **38.10** | **19.01** |

### Model Capacity
| Model | Params (M) | Relative to Teacher |
|:--|--:|--:|
| ResNet‑34 (Teacher) | **21.336** | 1.0× |
| Student‑50 (ResNet18_8x) | **11.220** | ~0.53× |
| Student‑20 (ResNet18_8x_Small) | **2.821** | ~0.13× |

**Protocol notes**
- Teacher: **ResNet‑34** (CIFAR‑100).  
- Distillation: KL on teacher/student logits over **synthetic** images from a lightweight generator (no original training set).  
- Training schedule: **15× student steps** followed by **1× generator step** (mirrors original notebook dynamics).  
- Reproducibility: run metadata and `config.json` are written per‑run under `./logs/<run_id>/` with fixed seeds.

## Training & Evaluation
- Configure defaults in `akd_kd_gan/config.py`.  
- `scripts/train_akd.py` builds the teacher, trains **Student‑50** and **Student‑20** alternating with the generator, evaluates after each epoch, and saves best checkpoints to `./models/<run_id>/`.

## References
- Hinton et al., Knowledge Distillation (2015)  
- Micaelli & Storkey, Zero‑Shot Knowledge Transfer via Adversarial Belief Matching (2019)
