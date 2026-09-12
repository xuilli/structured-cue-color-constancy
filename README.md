**Structured Representation and Interaction of Chromaticity Cues for Color Constancy in Pure Color Images**.

This implementation uses [PyTorch](http://pytorch.org/).

### Preparation

Python version >=3.9 is required.

### Dataset

The PolyU Pure Color Constancy Dataset V2 can be downloaded from [here](https://1drv.ms/u/c/73ea3202965da3e5/EdiFeWl15_lKrSfoeilDu8cBiIgizFL5MrMVUjSHZKctyA?e=RSEN4C).

#### Install dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Configuration

All settings are defined in `config/params.py`. Before running, set the dataset path:

```python
data_root: Path = Path("/path/to/pure_color_v2")
```

Common options:

- `experiment_name`: name of the experiment output folder.
- `epochs`: number of training epochs.
- `batch_size`: training batch size.
- `learning_rate`: initial learning rate.
- `run_all_folds`: whether to run all folds; defaults to `True`.
- `fold_index`: fold to run when `run_all_folds=False`.

Model settings are also available in `config/params.py`.

### Training

Run from the repository root:

```bash
python train.py
```

The default configuration performs three-fold cross-validation. The best checkpoint for each fold is saved to:

```text
outputs/<experiment_name>/fold_<index>/best.pth
```

### Evaluation

After training, run:

```bash
python test.py
```

Keep the configuration consistent with training. The script loads the best checkpoint for each fold and reports Mean, Median, Trimean, Best 25%, and Worst 25% angular errors.

Evaluation results are saved under `outputs/<experiment_name>/`.
