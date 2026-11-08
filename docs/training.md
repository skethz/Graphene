# Training recipe

The six `graphene-classical` and `graphene-lightweight` configurations implement the main manuscript's 500,000-step DF2K schedule. The dataset is supplied by the user; no training or testing images are included in this repository.

The crop size is explicitly an **LR** size. A 128×128 LR patch is paired with a scale-times-larger HR crop. Random crops, horizontal flips, and quarter-turn rotations are shared between the pair. Flips combined with the four rotations cover the eight square symmetries.

`tools.train.optimizer_partition` allocates the weights of `nn.Linear` modules under the shared encoder's `layers` to Muon. This includes attention projections, ConvFFN projections, and the IRB unary, initial-state, and residual maps. Convolutions, biases, LayerNorm parameters, relative-position tables, and HistAttn parameters use Adam. Every trainable parameter belongs to exactly one optimizer. Both optimizers start at 2×10⁻⁴ and receive the same milestone schedule.

BF16 autocast is used for encoder kernels. Parameters, optimizer states, the shallow feature path, and reconstruction/output layers remain FP32. EMA weights use decay 0.999. EMA decay and Muon's momentum 0.95, Nesterov setting, and five Newton–Schulz iterations are explicit implementation settings in the configuration.

Global batch size is always 32. With 16 processes, each process receives two samples per optimizer step. On fewer GPUs, `--micro-batch` controls the local forward batch and the trainer accumulates the remaining gradients. Unsupported divisions are rejected instead of rounding the global batch. Milestones count optimizer steps, not microbatches.

Training starts from random initialization. `--resume` restores weights, EMA, both optimizer states, and the optimizer-step count from this trainer. Data-worker augmentation streams are restarted when resuming. Reference inference checkpoints contain model weights and are not full training-resume snapshots.

```bash
python -m unittest tests.test_recipe -v
```

The tests check the recipe, global-batch accounting, paired crops, optimizer allocation, missing-weight rejection, window padding, and the IRB recurrence. `--max-steps` stops a training command early for a functional check without changing its learning-rate schedule.

## Real-world SR

The official [SwinIR repository](https://github.com/JingyunLiang/SwinIR#training) links to KAIR for training. The `graphene-realworld` configurations adapt KAIR's [PSNR stage](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/options/swinir/train_swinir_sr_realworld_x4_psnr.json) and [GAN stage](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/options/swinir/train_swinir_sr_realworld_x4_gan.json). Both retain the Graphene-realworld architecture and its nearest-neighbor/convolution ×4 reconstruction head.

| Setting | PSNR pretraining | GAN fine-tuning |
| --- | --- | --- |
| Data | DIV2K + Flickr2K + OST | DIV2K + Flickr2K + OST |
| Degradation | Original BSRGAN | Original BSRGAN |
| LR / HR crop | 64×64 / 256×256 | 64×64 / 256×256 |
| Global batch | 32 | 32 |
| Optimizer | Adam, β=(0.9, 0.999) | Adam for G and D, β=(0.9, 0.999) |
| Initial learning rate | 2×10⁻⁴ | 1×10⁻⁴ for G and D |
| LR halving milestones | 500k, 800k, 900k, 950k, 1M | 400k, 500k, 550k, 575k, 600k |
| Losses | L1 | L1 + VGG19 perceptual + adversarial |
| Generator loss weights | 1 | 1 / 1 / 0.1 |
| Precision / EMA | FP32 / 0.999 | FP32 / 0.999 |

The upstream configurations list learning-rate milestones but use an open-ended training loop. The release sets explicit stopping points at the last listed milestones: 1,000,000 PSNR steps and 600,000 GAN steps. Milestones count optimizer updates and apply to both optimizers in the GAN stage.

HR inputs must be RGB images of at least 256 pixels on each side. Image directories and encoded-image LMDBs are supported. The data loader draws 256×256 crops, applies the eight flip/rotation augmentations, and uses BSRGAN to form 64×64 LR inputs. As in KAIR, filenames containing `face` receive horizontal-flip augmentation only. Undersized images raise an explicit error. Datasets are supplied separately.

The VGG19 perceptual loss uses ImageNet-pretrained features at layers 2, 7, 16, 25, and 34, with weights 0.1, 0.1, 1, 1, and 1 and ImageNet input normalization. TorchVision downloads these frozen feature-extractor weights on first use if they are not already cached. The discriminator uses 64 base channels, spectral normalization, and orthogonal initialization with gain 0.2. Its loss is the sum of real and fake binary cross-entropies, as in KAIR.

`--initialize` supplies PSNR-stage generator/EMA weights for GAN fine-tuning. `--resume` restores a full snapshot from the same stage, including the generator, EMA, discriminator, optimizers, and iteration. Reference inference weights remain separate from training snapshots. Data-worker augmentation streams restart on resume.

```bash
python -m unittest tests.test_recipe tests.test_real_recipe -v
```
