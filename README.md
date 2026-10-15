# Telecom Churn: Imbalance-Aware Models and GP-Stacked Ensembles

Which telecom customers are about to leave, and does a stacked ensemble with an evolved meta-learner beat a single gradient-boosted model? A churn study on 7,043 customers (26.5% churn) with every preprocessing step inside scikit-learn pipelines.

The full analysis, with outputs, is in [`churn_stacking.ipynb`](churn_stacking.ipynb).

## Results on the held-out test set (20%, stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Stacking, genetic-programming meta-learner | 0.757 | 0.529 | 0.789 | 0.633 | **0.850** |
| Stacking, logistic meta-learner | 0.766 | 0.541 | 0.783 | **0.640** | 0.849 |
| XGBoost (class-weighted) | 0.762 | 0.536 | 0.786 | 0.637 | 0.849 |
| Random forest | **0.798** | **0.654** | 0.505 | 0.570 | 0.847 |
| Balanced random forest | 0.765 | 0.540 | 0.781 | 0.638 | 0.846 |
| Under-sampling + random forest | 0.752 | 0.521 | 0.789 | 0.628 | 0.845 |
| Logistic regression (class-weighted) | 0.754 | 0.525 | **0.794** | 0.632 | 0.845 |

- Ranking quality is saturated at ROC AUC of about 0.85; the top 20% of scored customers contains about half of all churners for every model.
- Class re-weighting lifts recall from 51% to about 79%.
- The evolved meta-learner reduces to a near-linear blend of the base models (about 2.1 × random forest + 3.1 × XGBoost). McNemar's test shows its errors differ from the logistic stack's (p = 0.006) without improving on it.
- SHAP ranks contract type, tenure and fibre-optic internet as the strongest churn drivers.

## Method

- **Preprocessing** in a `ColumnTransformer`: standardised numeric features, ordinal contract length, one-hot categoricals; fitted on the training split only.
- **Models:** logistic regression, random forest, balanced random forest, random under-sampling + random forest, XGBoost with `scale_pos_weight`, and two `StackingClassifier`s over random forest and XGBoost, one with a logistic meta-learner and one with a `gplearn` `SymbolicClassifier` evolved by genetic programming.
- **Evaluation:** accuracy, precision, recall, F1, ROC AUC and Brier score on the test set; cumulative gains; calibration curves; SHAP on XGBoost; McNemar's test between the two best models.

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute churn_stacking.ipynb --output executed.ipynb
```

The data is downloaded on first run. The GitHub Actions workflow executes the notebook on every push.

## Data

IBM [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) sample dataset, Apache-2.0.
