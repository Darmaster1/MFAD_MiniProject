# Predicting Student Final Grades via Orthogonal Projection and Least Squares

A linear algebra project that predicts a student's final grade (G3) from academic and lifestyle factors. The model is built, cleaned and fitted using only linear algebra (RREF, Gram-Schmidt/QR, orthogonal projection, least squares, eigenvalues). No machine-learning library is used for fitting.

## Dataset

UCI Student Performance dataset, Mathematics course file (`student-mat.csv`): 395 students, 33 columns, `;`-separated.

Source: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008. UCI Machine Learning Repository.

The bundled CSV was obtained from a public GitHub copy of the UCI file, with the expected shape of 395 × 33. To use the original, download it from UCI and replace `student-mat.csv`.

**Features used (7):** `studytime`, `failures`, `absences`, `G1`, `G2`, `freetime`, `goout`
**Target:** `G3` (final grade, 0 to 20)
**Engineered column:** `G_sum = G1 + G2`, added deliberately as an exactly dependent column so that the RREF and basis stages have a real redundancy to detect and drop.

## Files

| File | Description |
|---|---|
| `student_performance_single.ipynb` | Main notebook: the full pipeline in a single code cell, with outputs |
| `student_performance_single.py` | Same code as a plain Python script |
| `student-mat.csv` | The dataset (must be in the same folder as the code) |
| `viva_cheat_sheet.html` | Self-contained viva cheat-sheet (open in any browser) |

## Requirements

- Python 3.9+
- numpy, pandas, scipy, sympy, matplotlib
- jupyter (only for the notebook)

```bash
pip install numpy pandas scipy sympy matplotlib jupyter
```

## How to run

```bash
jupyter notebook student_performance_single.ipynb
```

Run the cell. Or, as a script:

```bash
python student_performance_single.py
```

Keep `student-mat.csv` in the same folder. The script opens two figure windows (results and eigenvalue spectrum).

## Method

The data is split 80/20 (seed 42): 316 training rows and 79 test rows. Columns are standardized with training statistics only, so no test information leaks into the fit. A column of 1s is prepended for the intercept.

| Stage | Concept | Purpose |
|---|---|---|
| 1 | Matrix representation | Turn the table into `Ax = b` |
| 2 | RREF, rank, nullity (SymPy, exact) | Find redundant columns; test whether `Ax = b` is consistent |
| 3 | Linear independence, basis | Keep the independent (pivot) columns |
| 4 | Gram-Schmidt (QR) | Orthonormalize the columns: `A = QR`, `QᵀQ = I` |
| 5 | Projection onto Col(A) | `p = QQᵀb`, residual `e = b − p`, check `Aᵀe ≈ 0` |
| 6 | Least squares | Solve `Rx̂ = Qᵀb` (back-substitution); compared with normal equations |
| 7 | Eigenvalues of `AᵀA` | Check conditioning and multicollinearity |

## Results

| Check | Value |
|---|---|
| rank(A) vs n (with `G_sum`) | 8 < 9, null-space vector (G1: −1, G2: −1, G_sum: +1) |
| rank([A\|b]) | 9 > rank(A), so the system is inconsistent |
| max abs(QᵀQ − I) | 3.6e-15 |
| max abs(Aᵀe), train | 2.8e-13 |
| cond(AᵀA), after dropping `G_sum` | 13.9 |
| Train RMSE / R² | 1.888 / 0.827 |
| **Test RMSE / R²** | **1.906 / 0.822** |

G2 dominates the model (standardized coefficient 3.64, next largest is G1 at 0.48).

## Notes and limitations

- G2 alone gives a test R² of about 0.826 (RMSE 1.881), slightly better than the 7-feature model. The additional features add essentially nothing out of sample.
- G1 and G2 are correlated (r = 0.84) but not dependent. The eigenvalue analysis shows only mild multicollinearity.
- `Aᵀe ≈ 0` holds on the training data, where the fit is made. It does not hold on the test rows, which is expected.
- Coefficients are conditional on G1 and G2 and should not be read causally. For example, a positive `absences` coefficient does not mean absences help grades.
- Students with G3 = 0 (dropouts) cannot be predicted by a linear model and inflate the error.
