# Predictive Modeling Pipelines using Tidymodels in R

This repository contains two comprehensive predictive modeling tasks developed using the modern `tidymodels` framework in R. The projects demonstrate best practices in machine learning, including stratified data splitting, robust feature engineering, ensemble modeling, parallel processing, and automated hyperparameter tuning.

---

## Repository Structure

* `validation task/`
  * `.gitignore`
  * `README.md`
  * `01_biomass_walidacja_modeli.qmd` -> Task 1: Validation and Model Comparison
  * `02_hotel_rates_tuning.qmd` -> Task 2: Feature Engineering and Simulated Annealing

---

## Project Overview

### Task 1: Biomass High Heating Value (HHV) Prediction
**Objective:** Compare the performance of four distinct machine learning architectures to predict the High Heating Value (HHV) of biomass samples based on their elemental composition.

* **Dataset:** `biomass` (from the `modeldata` package).
* **Key Engineering Steps:**
    * Removal of high-cardinality structural identifiers (`sample` and `dataset`) to eliminate target leakage.
    * Stratified data splitting (80/20 train/test ratio) to maintain uniform distribution across segments.
    * **Custom Recipe Strategies:**
        * *Linear Regression:* Addressed skewed features using a `Yeo-Johnson` transformation followed by standard normalization.
        * *Support Vector Machines (SVM):* Applied Principal Component Analysis (`PCA`) retaining 95% of total variance to resolve predictor collinearity.
        * *Tree-Based Models (MARS & Random Forest):* Modeled using robust scale-invariant raw representations.
* **Validation & Statistical Rigor:**
    * Evaluated models across 5-fold cross-validation with 2 repeats.
    * Conducted a **Paired t-test** ($p = 0.025$) comparing the top-performing architectures (Random Forest vs. MARS) to statistically justify the final model selection.

### Task 2: Advanced Hotel Room Rates Forecasting & Optimization
**Objective:** Predict average daily hotel room prices (`avg_price_per_room`) while managing complex temporal indicators and sparse categorical groupings.

* **Dataset:** `hotel_rates` (from the `modeldata` package), downsampled to 1,500 samples for computational efficiency during global heuristic searches.
* **Advanced Feature Engineering:**
    * Deconstructed date-based entries (`arrival_date`) into explicit numeric and factor vectors (Day of Week, Month, Year).
    * Grouped rare categorical levels below a 2% threshold into a unified `"Other"` bin via `step_other` to prevent sparse matrix bloat.
    * Engineered a new domain-specific interaction term: `total_nights` (combining weekday and weekend night stays).
* **Screening and Heuristic Tuning:**
    * **Screening Phase:** Evaluated Elastic Net, Random Forest, and XGBoost baselines over shared folds using a `workflow_set`.
    * **Global Optimization:** Selected the best architecture (**XGBoost**) and optimized its hyperparameters (`trees`, `tree_depth`, and `learn_rate`) utilizing a **Simulated Annealing** algorithm (`finetune::tune_sim_anneal`).
    * **Performance Acceleration:** Enabled asynchronous multi-core parallel processing (`doParallel`) to distribute resampling routines across available CPU threads.
* **Overfitting Assessment:** Discussed the generalization gaps by closely mapping cross-validation metrics against final performance metrics on the unseen test holdout set.

---

## Dependencies & Setup

Ensure you have R and the required library stack installed before attempting to render the Quarto documents:

```r
install.packages(c("tidymodels", "tidyverse", "finetune", "doParallel", "vip", "earth", "kernlab", "ranger", "xgboost", "glmnet"))
