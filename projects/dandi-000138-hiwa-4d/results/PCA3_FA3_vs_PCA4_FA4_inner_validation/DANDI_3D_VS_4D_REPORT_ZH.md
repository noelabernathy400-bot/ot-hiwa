# DANDI 3D vs archived 4D: real neural–hand matching and OT errors

**Same raw NWB SHA and same trial-disjoint 320/80 internal split; original 100 test trials never used.**
3D: neural 162ch -> train PCA3 or train FA3; target original hand pos_x,pos_y,vel_x,vel_y -> train-standardize -> train PCA3. 4D archival comparator: neural PCA4 or FA4, original 4-component target. For each method, K4,40/group, original 500/20/45/200, mu.05,eps_s.1,eps_g.2, same seed 42-44 and same independent domain grouping algorithms. No paired labels go into fitting.

**Primary comparison on exactly identical original 4D physical movement**: cross-trial position and velocity vector mean error, position hit <=10% train median trial path, velocity angle <=30deg (train speed exclusion). All nearest neighbours chosen only after fit and exclude the source window's trial.

| Source / target | Group | N seeds | Free W2² (native d) | Relative drop vs same-d identity | Nested P/Q (native d) | Position mean error (native) | Velocity vector mean error (native) | Position hit10% | Direction hit30deg | Strict ADMM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA3 / handPCA3 | Hard | 3 | 0.3611 | -6.31% | 1.1675 | 110.5284 | 466.9144 | 9.35% | 15.78% | 3/3 |
| PCA3 / handPCA3 | Soft | 3 | 0.3312 | 2.49% | 1.4482 | 109.2203 | 448.6033 | 13.50% | 18.35% | 2/3 |
| FA3 / handPCA3 | Hard | 3 | 0.2969 | 14.78% | 1.0545 | 112.7151 | 439.2707 | 9.73% | 21.52% | 3/3 |
| FA3 / handPCA3 | Soft | 3 | 0.2783 | 20.14% | 1.4449 | 110.9886 | 455.4689 | 10.55% | 21.94% | 3/3 |
| PCA4 / handoriginal4 | Hard | 3 | 0.4156 | 10.95% | 1.3491 | 127.3590 | 495.9752 | 6.42% | 13.33% | 3/3 |
| PCA4 / handoriginal4 | Soft | 3 | 0.3774 | 19.13% | 1.7618 | 147.7242 | 499.2850 | 6.63% | 14.29% | 0/3 |
| FA4 / handoriginal4 | Hard | 3 | 0.3603 | 15.85% | 1.3427 | 133.7742 | 455.5428 | 9.06% | 18.02% | 3/3 |
| FA4 / handoriginal4 | Soft | 3 | 0.4078 | 4.76% | 1.7711 | 141.9579 | 481.5800 | 8.33% | 15.57% | 0/3 |

Same physical random cross-trial baseline: position hit10%=10.00%; direction hit30deg=13.20%.
Train median path 10% physical position radius = 18.457 original stored units (do not invent cm/mm).

**Critical interpretation:** native latent free W2² and groupwise nested P/Q transport use different 3D versus 4D feature spaces. The absolute numbers cannot alone determine which embedding is better. Within-dim relative OT improvement and matching errors evaluated in ORIGINAL physical4D coordinates are more comparable.
Hard versus Soft native nested OT costs use different group probability measures; do not rank Soft solely by these unlike objective values.
Numerical Soft ADMM failures must be reported; not optimized means no unconditional win claim.
This is a **single internal 320/80 split with 3 optimizer seeds**, not an independent final confirmation. Do not use the previously inspected original 100-test trials to select 3D versus 4D.

[Full JSON with all seeds](DANDI_3D_VS_4D_RESULTS.json) | [3D full P/Q/R blocks](.) | [Underlying archived 4D results](../PCA4_FA4_inner_validation/DANDI_PCA4_FA4_REPORT_ZH.md)
