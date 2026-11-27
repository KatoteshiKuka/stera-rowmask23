# Weights

The weights are not versioned on GitHub. On the author's machine this folder contains:

| File | Content | SHA-256 |
|---|---|---|
| `stera30m_step122000.ckpt.gz` | Final STERA 30M checkpoint (122,000 updates), with compact weights and optimizer state | `418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887` |

To obtain it:
- download it from the repository Release `v1.0` (file `stera30m_step122000.ckpt.gz`) into this folder:
  `gh release download v1.0 --pattern stera30m_step122000.ckpt.gz --dir weights`
- or rerun `training/train_stera30m.sh` (about 35 hours on two P100s) and copy `TS68V2_STEP122000.ckpt.gz` here under this name.

The scripts verify the hash before loading the file.
