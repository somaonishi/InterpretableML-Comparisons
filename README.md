# Why Do Tree Ensemble Approximators Not Outperform the Recursive-Rule eXtraction Algorithm?
  
🎉 Accepted at Machine Learning and Knowledge Extraction ([paper](https://www.mdpi.com/2504-4990/6/1/31)). 🎉

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

```bib
@article{make6010031,
	author = {Onishi, Soma and Nishimura, Masahiro and Fujimura, Ryota and Hayashi, Yoichi},
	doi = {10.3390/make6010031},
	issn = {2504-4990},
	journal = {Machine Learning and Knowledge Extraction},
	number = {1},
	pages = {658--678},
	title = {Why Do Tree Ensemble Approximators Not Outperform the Recursive-Rule eXtraction Algorithm?},
	url = {https://www.mdpi.com/2504-4990/6/1/31},
	volume = {6},
	year = {2024},
	bdsk-url-1 = {https://www.mdpi.com/2504-4990/6/1/31},
	bdsk-url-2 = {https://doi.org/10.3390/make6010031}
}
```

## Thanks
Special thanks to collaborators, [Masahiro Nishimura](https://github.com/nishimura28) and [Ryota Fujimura](https://github.com/fuji12345), for their invaluable contributions to this project.
