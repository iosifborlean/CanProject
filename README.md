# Project Documentation

## Table of Contents

- [1. Pipeline Orchestrator (`main.py`)](#1-pipeline-orchestrator-mainpy)
  - [1.1 Hyperparameter Fine-Tuning & Model Selection](#11-hyperparameter-fine-tuning--model-selection)
  - [1.2 Feature Analysis](#12-feature-analysis)
  - [1.3 Folds & Training Splits Analysis](#13-folds--training-splits-analysis)
  - [1.4 Recording Times Evaluation](#14-recording-times-evaluation)
  - [1.5 Single Board Performance](#15-single-board-performance)
  - [1.6 Downsampling Performance](#16-downsampling-performance)
  - [1.7 Downsampling + Feature Analysis](#17-downsampling--feature-analysis)
- [2. Signal Processing (`Saleae` to CSV)](#2-signal-processing-saleae-to-csv)
- [3. Dataset Preparation (`CSV` to Dataset)](#3-dataset-preparation-csv-to-dataset)

---

## 1. Pipeline Orchestrator (`main.py`)

`main.py` serves as the primary CLI entry point for executing experiments, optimization loops, and performance benchmarking scripts.

---

### 1.1 Hyperparameter Fine-Tuning & Model Selection

* **Entry Function:** `model_optimization_and_choice()`
* **Outputs:** `results/best_hyperparams.txt`, `results/best_model`

Systematically tunes and compares candidate classification models (`xgboost`, `random_forest`, and `svm`) to identify the top-performing architecture. 

For each algorithm, the routine invokes `optimize_hyperparams()` to execute a 5-fold cross-validated grid search (`GridSearchCV`) over predefined hyperparameter grids—incorporating pipeline-level preprocessing such as `StandardScaler` for SVMs and balanced sample weighting for XGBoost. 
After evaluating each tuned model on 35-second prototype recordings at 50 Hz, the function selects the best overall model architecture, appends optimization logs to `results/best_hyperparams.txt`, and exports the winning configuration to `results/best_model`.

---

### 1.2 Feature Analysis

* **Entry Function:** `evaluate_features()`
* **Outputs:** `results/features_results.txt`, `results/best_model` (updated)

Runs an automated feature selection routine to identify the most resilient feature combination for the optimal model setup. 

The underlying procedure (`test_feature_subsets`) samples randomized feature subsets across predefined size tiers (5 to 25 features) and evaluates each using 3-fold Stratified K-Fold cross-validation (macro precision). 
By analyzing feature frequency across the top 20% highest-scoring combinations, it derives a distilled "robust feature set" based on a consensus inclusion threshold ($\ge 65\%$). The routine logs a detailed statistical report to `results/features_results.txt` and automatically updates `results/best_model` with the newly optimized feature configuration.

---

### 1.3 Folds & Training Splits Analysis

* **Entry Function:** `evaluate_folds_and_splits()`
* **Outputs:** `results/folds_and_training_splits.txt` log file

Benchmarks model stability and training data volume sensitivity using the optimal model configuration on 35-second prototype recordings at 50 Hz. 

The routine performs a 5-fold Stratified K-Fold cross-validation (`evaluate_cv_folds`) to assess model variance, followed by a data quantity sensitivity analysis (`evaluate_train_val_splits`) testing stratified training splits ranging from 5% to 70%. 
Both evaluations compute macro precision metrics (mean $\pm$ standard deviation) across iterations and write formatted performance breakdowns to a results log file.

---

### 1.4 Recording Times Evaluation

* **Entry Function:** `evaluate_recording_times()`
* **Outputs:** `results/recording_times_<car_type>.txt`

Runs an automated experiment evaluating model performance as a function of recording window size (1–30 seconds). 

Upon selection of the target vehicle type (`prototype` vs. `car`), the routine retrieves the top-performing model configuration, processes dataset features at 50 Hz across nine distinct durations, and writes the performance evaluation logs to `results/recording_times_<car_type>.txt`.

---

### 1.5 Single Board Performance

* **Entry Function:** `evaluate_single_board_type()`
* **Outputs:** `results/single_board_results.txt`

Evaluates classification performance when trained and tested on data isolated by specific sensor board types (`Aurix`, `ST`, `Atmel`) rather than aggregated datasets. 

Operating on 35-second prototype recordings at 50 Hz, the function iterates through each board configuration in `combinazioni_board`. It then evaluates model test metrics independently for each hardware context and appends performance summaries to `results/single_board_results.txt`.

---

### 1.6 Downsampling Performance

* **Entry Function:** `evaluate_downsampling_performance()`
* **Outputs:** `results/downsampling_results.txt`

Benchmarks how classification accuracy degrades as the sensor sampling rate is reduced. 

Using 35-second prototype recordings and the optimal model configuration, the routine iterates through downsampling factors from 1 to 8—reducing the effective sampling rate from 50 Hz down to 6.25 Hz. 
For each rate, it extracts features, evaluates model test metrics, and appends the comparative performance log to `results/downsampling_results.txt`.

---

### 1.7 Downsampling + Feature Analysis

* **Entry Function:** `evaluate_downsampling_and_features_performance()`
* **Outputs:** `results/downsampling_and_features_results.txt`

Identifies features that maintain high predictive utility across varying sampling frequencies. 

Working on 35-second prototype recordings, the function iterates through downsampling factors (1, 2, 3, 4, and 8; reducing rates from 50 Hz to 6.25 Hz) and runs randomized feature subset testing (`test_feature_subsets`) at each resolution. 
By tracking feature frequency across the top 20% performing models across all sampling tiers, it computes global cross-sampling inclusion rates to reveal potentially frequency-invariant features, saving the consolidated report to `results/downsampling_and_features_results.txt`.

---

## 2. Signal Processing (`Saleae` to CSV)

* **Key Function:** `parse_logic_files()`
* **Inputs:** Raw Saleae captures (`analog.csv`, `digital.csv`, `log.csv`)
* **Outputs:** Extracted signal segment CSV files (`voltages/<sample_rate>/`)

Reads raw CAN bus files (`analog.csv`, `digital.csv`, and `log.csv`), cuts out the voltage signals for each frame, checks that the signal start and end points are clean, and saves the clips to CSV files.

---

## 3. Dataset Preparation (`CSV` to Dataset)

* **Key Function:** `load_or_extract_features()`
* **Helper Function:** `load_and_split_voltage_dataset()`
* **Outputs:** Binary cache files (`caches/`)

Loads voltage files for target CAN IDs, downsamples the signals, and splits the dataset 80/20 into train/test sets mapped to specific ECU labels. Extracted features and raw datasets are automatically cached to disk (using `joblib` and `pickle`) to avoid re-processing on future runs.
