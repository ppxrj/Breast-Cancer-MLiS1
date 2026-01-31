# Copilot Instructions for Breast-Cancer-MLiS1

## Project Overview

This is an academic machine learning research project classifying breast cancer tumors using a custom-built **Decision Tree** and **Logistic Regression** implementation from scratch (no Keras/TensorFlow). The goal is a 2-page Physical Review Letters–style paper with accompanying research-quality code.

**Key constraint**: All ML implementations must use only numpy/scipy—no neural network packages allowed.

## Project Architecture

### Data Pipeline
1. **Data Ingestion**: Uses UCI ML Repo (`fetch_ucirepo(id=17)`) to fetch the breast cancer dataset (569 samples, 30 features)
   - Target encoding: 2→0 (benign), 4→1 (malignant)
   - Train/test split: 80/20 (seed=42) with shuffled indices

2. **Data Exploration** (`modules/stats_graph.py`):
   - Boxplots for triple feature measurements
   - Scatter plots and histograms for outlier detection
   - KDE, mean/median/±3σ visualization

3. **Feature Engineering**: 30 features derived from 10 measurement types (radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension), each measured 3 times

### ML Models (Custom Implementations from Scratch)

**Decision Tree** (`modules/Decision_Tree.py`):
- CART algorithm with entropy-based splitting (Breiman 1984, Quinlan 1986)
- Information gain: `IG = H(parent) - weighted_avg(H(children))`
- Stopping criteria: max_depth=10 (default), min_samples_split=2
- Greedy threshold search across all features
- Methods: `fit()` → `_grow_tree()` → `_best_split()` → `_information_gain()` → `predict()`

**Logistic Regression** (in notebooks, e.g., `Python/Ramona scribble.ipynb` cell 6):
- Cross-entropy loss with L2 regularization
- Sigmoid activation: σ(z) = 1/(1+e^(-z))
- Gradient descent optimization with convergence tracking
- Methods: `fit()` → gradient updates, `predict_proba()`, `predict(threshold=0.5)`

### Notebook Workflow
- **Main Code.ipynb** (stub): Intended orchestration point—needs imports and module calls
- **Python/Ramona scribble.ipynb** (active): Development notebook with LogisticRegression implementation and exploratory analysis
- **Python/Fixed_code.ipynb, Kam_Updated_Code.ipynb, No_graphs_code.ipynb**: Parallel experimental branches

### Report Structure
- **LaTeX_sections/**: Main.tex, Introduction.tex, Conclusion.tex
- **references.bib**: Bibliography
- **MLiS1 Report.tex/bbl**: Generated report artifacts
- Target format: 2-page PRL-style (RevTeX compatible)

## Critical Workflows

### Running the Project
1. Activate venv: `.\venv\Scripts\activate` (Windows)
2. Install dependencies: `pip install -r requirements.txt`
   - Key packages: numpy, pandas, matplotlib, scipy, seaborn, scikit-learn, ucimlrepo
3. Execute notebooks in order:
   - Data ingestion and EDA (stats_graph.py visualizations)
   - Train models (DecisionTree, LogisticRegression)
   - Evaluation and paper generation

### Key Data Structures
- **X**: DataFrame (569, 30) with 30 continuous features
- **y**: Series (569,) with target {0, 1} after encoding
- **X_train, X_test**: numpy arrays for model training (456, 113 samples)
- **y_train, y_test**: 1D numpy arrays (ravel required for some operations)

## Code Conventions & Patterns

### Model Implementation Patterns
- **Initialization**: Parameters (learning_rate, max_depth, etc.) set in `__init__()` with defaults
- **Training**: Single `fit(X, y)` method. Expect X as numpy array or DataFrame rows
- **Prediction**: Separate `predict()` (class labels) and `predict_proba()` (probabilities) methods
- **Numerical stability**: Clip sigmoid inputs to [-500, 500]; add ε=1e-9 to log operations
- **Loss tracking**: `self.losses` list for convergence monitoring (appended every 100 iterations)

### Data Handling
- Use `.iloc[]` for positional indexing (not `.loc[]`) when splitting
- `np.unique(y, return_counts=True)` for class distribution checks
- `np.random.seed(42)` for reproducibility
- `np.clip()` for preventing overflow in exponential operations

### Visualization Patterns (stats_graph.py)
- Feature lists hardcoded: `['radius','texture', 'perimeter', ...]`
- Triple measurements: Loop over features, then suffixes [1, 2, 3]
- Grid plots: `plt.subplots(rows, cols)` with `.flatten()` for iteration

## Integration Points & External Dependencies

- **ucimlrepo**: Fetches dataset dynamically. Import: `from ucimlrepo import fetch_ucirepo`
- **numpy**: Core numerical operations; models operate on numpy arrays
- **pandas**: Data loading and initial exploration; must convert to numpy for training
- **scipy.stats**: Normal distribution fitting and statistical tests
- **seaborn/matplotlib**: Visualizations (kde=True, stat='density' for distributions)
- **scikit-learn**: Available but used only for utilities (e.g., metrics); no Estimator base class usage

## Testing & Validation

- **Train/test performance**: Compare model predictions on X_test vs y_test
- **Convergence**: Check `self.losses` for monotonic decrease
- **Class balance**: Verify with `y_encoded.value_counts()` (~63% benign, ~37% malignant)
- **No validation set**: All eval happens on held-out test set

## Next Steps & Open Tasks

(From README and ltodolist.txt context):
- Consolidate scattered notebooks into unified Main Code.ipynb
- Run final model training and evaluation
- Generate figures (decision tree visualizations, confusion matrices) for the paper
- Write LaTeX sections synthesizing results
- Deadline: February 2nd 2026, 3pm

---

**For AI agents**: When adding features or fixing bugs, maintain the "from scratch" principle—implement algorithms using numpy operations, not library calls. Keep notebook cells small and document assumptions about data shape/encoding. Reference scientific papers (Breiman 1984, Quinlan 1986, Shannon 1948) in docstrings where applicable.
