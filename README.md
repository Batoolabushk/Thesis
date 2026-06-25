# Assessing the Reliability of a Non-Invasive, Lifestyle-Based Diabetes Risk Model

This repository contains the complete analysis pipeline, code, results, and figures for a reliability-aware machine learning study on non-invasive diabetes risk screening using CDC BRFSS survey data.

The study develops a calibrated XGBoost classifier trained on ten self-reportable lifestyle and demographic variables, and evaluates it not only on aggregate discrimination but on **subgroup-stratified reliability** across age, BMI, and sex — revealing that a model with acceptable overall performance can simultaneously fail specific population subgroups.

This work was conducted as part of a Bachelor's thesis at the University of Europe for Applied Sciences and a parallel manuscript prepared for submission to the *Journal of Medical Systems* (Springer Nature).

---

## Key Findings

- Calibrated XGBoost achieved **ROC-AUC 0.781** on the internal balanced test set and **0.772** on a BRFSS 2021 temporal validation cohort.
- At the recall-optimised screening threshold (0.35), recall reached **89.6%** with a false-negative rate of **10.4%**.
- Subgroup analysis revealed false-negative rates ranging from **86.0%** (age 18–34) to **13.2%** (age 65–74), and from **69.5%** (normal weight) to **5.8%** (obese class II/III) — exposing a reliability gradient hidden by aggregate metrics.
- Calibration collapsed under natural prevalence conditions (ECE 0.009 → 0.276), confirming that probability scores require prevalence-adjusted recalibration before deployment.
- Decision-curve analysis confirmed positive net benefit over treat-all for threshold probabilities of **0.01–0.17** at natural prevalence.
- Findings were synthesised into a four-tier deployment-readiness scorecard for responsible first-stage screening guidance.

---

## Repository Structure

```
.
├── notebook.ipynb              # Full analysis pipeline (single notebook, organised by section)
├── figures/                        # All publication-ready figures (PDF, vector, 300+ ppi)
│   ├── rq1_overall_performance_panel.pdf
│   ├── rq1_confusion_matrix_default.pdf
│   ├── rq1_cv_model_comparison_with_wilcoxon.pdf
│   ├── threshold_selection_curve_calibrated_validation.pdf
│   ├── rq2_feature_importance_panel.pdf
│   ├── rq3_rq5_subgroup_reliability_panel.pdf
│   ├── rq6_false_negative_heatmap_age_bmi.pdf
│   ├── rq6_umap_error_projection.pdf
│   ├── rq7_deployment_readiness_matrix.pdf
│   ├── calibration_analysis_multi_dataset.pdf
│   ├── decision_curve_PRIMARY_natural_prevalence.pdf
│   ├── decision_curve_SUPPLEMENTARY_balanced_internal_test.pdf
│   └── multi_seed_stability_panel.pdf
├── tables/                         # All result tables as CSV (40+ files)
│   ├── rq1_*.csv                   # Overall performance, model comparison, Wilcoxon tests
│   ├── rq2_*.csv                   # Permutation importance, SHAP, feature group ablation
│   ├── rq3_age_subgroup_reliability_metrics.csv
│   ├── rq4_bmi_subgroup_reliability_metrics.csv
│   ├── subgroup_metrics_external_temporal.csv
│   ├── rq6_*.csv                   # Error profiling, false-negative heatmap data
│   ├── rq7_*.csv                   # Deployment-readiness scorecard
│   ├── calibration_curve_*.csv
│   ├── decision_curve_*.csv
│   └── validation_and_prevalence_shift_evaluation.csv
├── models/
│   ├── best_calibrated_model.joblib   # Final calibrated XGBoost model
│   └── model_metadata.json            # Model configuration, features, thresholds
├── run_manifest.json                # Full run configuration and artifact index
├── interpretation_notes_for_manuscript.txt   # Internal QA notes used during manuscript writing
├── threshold_clinical_rationale.txt
└── README.md
```

---

## Data Sources

This study uses three publicly available CDC Behavioral Risk Factor Surveillance System (BRFSS) datasets. **Raw data files are not included in this repository** due to size; they must be downloaded from the original sources below.

| Dataset | Source | Role |
|---|---|---|
| BRFSS 2015 (balanced 50/50) | [Kaggle — CDC Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | Training and internal test set ($n = 70{,}692$) |
| BRFSS 2015 (full, natural prevalence) | [CDC BRFSS Annual Data — 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html) | Calibration and decision-curve analysis ($n = 253{,}680$) |
| BRFSS 2021 | [CDC BRFSS Annual Data — 2021](https://www.cdc.gov/brfss/annual_data/annual_2021.html) | Temporal validation cohort ($n = 230{,}759$) |

All datasets are public, anonymised, and contain no patient-identifiable information.

---

## Methodology Summary

1. **Feature selection** — ten non-invasive, self-reportable variables: BMI, physical activity, smoking, fruit/vegetable consumption, heavy alcohol use, sex, age, education, and income. All clinical/laboratory variables were excluded.
2. **Model selection** — seven classifiers (XGBoost, Gradient Boosting, HistGradientBoosting, Random Forest, Logistic Regression, Extra Trees, Decision Tree) compared via 5-fold stratified cross-validation. XGBoost selected on highest mean ROC-AUC (0.778).
3. **Calibration** — isotonic regression applied via `CalibratedClassifierCV` (5-fold), evaluated at three operating thresholds: default (0.50), Youden-J (0.53), and recall-optimised screening (0.35).
4. **Reliability evaluation** — subgroup-stratified false-negative rate analysis across age, BMI, and sex; error profiling via confusion-matrix breakdown and UMAP projection; calibration assessment at natural prevalence; decision-curve analysis; temporal validation on a 2021 cohort.
5. **Deployment synthesis** — a four-tier reliability scorecard (higher / moderate / low-moderate / low) classifying each subgroup's suitability for standalone screening.

Full methodological detail is provided in the accompanying thesis and manuscript.

---

## Reproducing the Results

### Requirements

```
python>=3.11
scikit-learn==1.4
xgboost==2.0
pandas==2.2
numpy==1.26
matplotlib==3.8
seaborn==0.13
shap==0.45
umap-learn==0.5
scipy
joblib
```

Install via:

```bash
pip install scikit-learn==1.4 xgboost==2.0 pandas==2.2 numpy==1.26 matplotlib==3.8 seaborn==0.13 shap==0.45 umap-learn==0.5 scipy joblib
```

### Running the analysis

1. Download the three BRFSS datasets listed above and place them in a `data/` directory.
2. Open `notebooks/notebook.ipynb` and run all cells sequentially.
3. All figures, tables, and the trained model will be written to `figures/`, `tables/`, and `models/` respectively.

A fixed random seed (`42`) is used throughout all stochastic operations — train-test splitting, cross-validation, model training, UMAP initialisation, and bootstrap resampling — to ensure full reproducibility. Multi-seed stability is additionally verified across five seeds (11, 42, 77, 101, 202).

No GPU is required; the full pipeline runs on a standard CPU environment.

---

## Important Notes on Interpretation

- **This is not a diagnostic tool.** The model is evaluated and presented strictly as a first-stage, non-clinical screening instrument. It does not replace blood glucose testing, HbA1c measurement, or clinical assessment.
- **The deployment-readiness scorecard is an author-defined research framework** based on statistical thresholds (recall, false-negative rate, ROC-AUC) selected for research purposes. It has not been prospectively validated in clinical settings and should not be used as clinical guidance without further validation.
- **Temporal validation on BRFSS 2021 is same-source, not independent external validation** — the 2021 cohort shares the same survey methodology and feature coding as the 2015 training data.
- **Probability scores require prevalence-adjusted recalibration** before being interpreted as absolute risk estimates outside the balanced training distribution; see the calibration analysis in `tables/validation_and_prevalence_shift_evaluation.csv` for details.

---

## Citation

If you use this code or findings, please cite:

```
Abushkaidem, B.A.R., Ali, R.H., Ahmed, I. (2026). Assessing the Reliability of a
Non-Invasive, Lifestyle-Based Diabetes Risk Model. [Manuscript in preparation,
Journal of Medical Systems].
```

---

## Authors

- **Batool A.R. Abushkaidem** — University of Europe for Applied Sciences
- **Raja Hashim Ali** — Supervisor
- **Iftikhar Ahmed** — Second Supervisor

---

## License

This repository is shared for academic and research purposes. Please contact the author before any commercial use.
