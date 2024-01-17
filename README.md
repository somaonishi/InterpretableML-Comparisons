# InterpretableML Comparisons
## Environment
- python: >=3.10,<3.11

## Setup
Initialize submodules.

```bash
git submodule update --init --recursive
```

We use [poetry](https://python-poetry.org/) to manage dependencies.
See https://python-poetry.org/ for usage.

```bash
poetry install
```

Or you can use pip.

```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

We use [hydra](https://hydra.cc/) to manage configurations.
See https://hydra.cc/ for usage.

**Change dataset**
```bash
python main.py data=base data.name=Occupancy
```

**Change model**
```bash
python main.py model=dt
```

**Run with optuna**
```bash
python main.py exp=optuna
```

### How to replicate our experiment?
```bash
python script/run_all.py 40981,44096,53,40715,50,44,1462,1494,1558,45557,Occupancy
```

## Citation
under review

## Thanks
Special thanks to collaborators, [Masahiro Nishimura](https://github.com/nishimura28) and [Ryota Fujimura](https://github.com/fuji12345), for their invaluable contributions to this project.
