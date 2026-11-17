"""Deep Survival Machines (Nagpal et al., IEEE JBHI 2021).

dsm_torch.py, losses.py and utilities.py are copied from the official implementation,
auton-survival (https://github.com/autonlab/auton-survival, auton_survival/models/dsm/),
MIT License (see LICENSE in this folder). The package itself pins torch<2 and numpy<2, so it
does not install next to our environment. The only change is that tqdm is optional.
The wrapper we use is run_dsm in src/baselines.py.
"""
