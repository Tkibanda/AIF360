# Reweighing (Kamiran & Calders, 2012) on Adult, German Credit and COMPAS

Tonny Kibanda · Student no. 2160579

This folder was added to a fork of [AIF360](https://github.com/Trusted-AI/AIF360) for a bias-mitigation assignment.
It evaluates AIF360's `Reweighing` pre-processing algorithm on three datasets, using the AIF360 library code as is.

| File | Content |
|---|---|
| `Task2_Reweighing.ipynb` | Full pipeline, already run, with all outputs saved |
| `requirements.txt` | Exact package versions used |

## Exact steps to reproduce

Tested with Python 3.12.

1. Clone this repository and go to this folder:
   ```
   git clone https://github.com/Tkibanda/AIF360.git
   cd AIF360/reweighing_task2
   ```
2. Create a virtual environment and install the packages:
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Start Jupyter and open `Task2_Reweighing.ipynb`:
   ```
   jupyter notebook
   ```
4. Run all cells (**Kernel → Restart & Run All**). It takes a few minutes, mostly on Adult.

**Google Colab instead:** upload the notebook, uncomment `!pip install aif360` in the first cell, then **Runtime → Run all**.

## Data

No manual download is needed. The first code cell downloads the raw files into AIF360's `data/raw` folder:

| Dataset | Source | Files |
|---|---|---|
| Adult | UCI ML Repository | `adult.data`, `adult.test` |
| German Credit | UCI ML Repository (Statlog German Credit) | `german.data` |
| COMPAS | ProPublica `compas-analysis` repository | `compas-scores-two-years.csv` |

If a download fails, download the files by hand and place them in the folder the notebook prints after `data folder:`, in the subfolders `adult/`, `german/` and `compas/`.

## Settings used

| Setting | Value |
|---|---|
| Sensitive attribute (privileged vs unprivileged) | Adult: sex (male vs female); German: age (≥ 25 vs < 25); COMPAS: race (Caucasian vs other) |
| Train/test split | 70/30, stratified on label × sensitive group |
| Random seeds | 0, 1, 2, 3, 4 (results are means ± std) |
| Mitigation | AIF360 `Reweighing`, fitted on the training data only (no hyperparameters) |
| Classifiers | `StandardScaler` + `LogisticRegression(max_iter=2000)` (stable) and `DecisionTreeClassifier(min_samples_leaf=10)` (unstable); weights passed as `sample_weight` |
| Model inputs | All features except the sensitive attribute |
| Metrics | DP_Diff (statistical parity difference), DIR (disparate impact), consistency (k = 5, standardised features), accuracy, balanced accuracy |

## Expected results (test set, means over 5 seeds)

| Dataset | Model | DP_Diff | DIR | Consistency | Accuracy |
|---|---|---|---|---|---|
| Adult | LogReg baseline | −0.185 | 0.303 | 0.944 | 0.845 |
| Adult | LogReg + Reweighing | −0.102 | 0.549 | 0.945 | 0.842 |
| Adult | Tree baseline | −0.176 | 0.329 | 0.903 | 0.847 |
| Adult | Tree + Reweighing | −0.105 | 0.560 | 0.900 | 0.841 |
| German | LogReg baseline | −0.136 | 0.824 | 0.814 | 0.749 |
| German | LogReg + Reweighing | −0.098 | 0.874 | 0.812 | 0.750 |
| German | Tree baseline | −0.076 | 0.905 | 0.762 | 0.721 |
| German | Tree + Reweighing | −0.075 | 0.902 | 0.757 | 0.717 |
| COMPAS | LogReg baseline | −0.144 | 0.795 | 0.858 | 0.667 |
| COMPAS | LogReg + Reweighing | −0.132 | 0.812 | 0.859 | 0.668 |
| COMPAS | Tree baseline | −0.146 | 0.791 | 0.753 | 0.658 |
| COMPAS | Tree + Reweighing | −0.142 | 0.793 | 0.751 | 0.649 |

## Notes

- On import, AIF360 warns about optional packages (TensorFlow, Fairlearn, inFairness) that Reweighing does not need; the notebook hides these notices.
- AIF360 0.6.1 does not work with pandas 3's default string type. The notebook switches it off with `pd.set_option("future.infer_string", False)`; on older pandas versions this line does nothing.
- AIF360's own `consistency()` is very slow on Adult, so the notebook uses the same formula with a fast brute-force nearest-neighbour search.
- DP_Diff follows AIF360's sign convention: P(favourable | unprivileged) − P(favourable | privileged). Negative values mean the unprivileged group is disadvantaged.
