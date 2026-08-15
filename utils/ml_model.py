from pathlib import Path

from sklearn.metrics import classification_report


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit
)

import random
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import numpy as np
import warnings

results_directory = Path(__file__).resolve().parents[1] / "results"
results_hyperparams_filepath = results_directory / "best_hyperparams.txt"
results_folds_and_training_splits_filepath = results_directory / "folds_and_training_splits.txt"


class MLmodel:
    ALL_FEATURES = [
        "v_mean", "v_max", "t_bit", "t_plat",
        "time_maximum", "time_minimum", "time_mean", "peak", "mean_absolute", "variance",
        "std_dev", "time_kurtosis", "skewness", "rms", "form_factor", "peak_factor",
        "impulse_factor", "crest_factor",
        "centroid", "freq_crest", "freq_peak", "freq_mean", "decrease", "entropy",
        "flatness", "arithmetic_mean", "geometric_mean", "flux", "freq_kurtosis",
        "rolloff_point", "freq_skewness", "slope", "spread"
    ]

    def __init__(self, sample_rate, ml_model, selected_features=None, hyperparams=None):
        self.model_type = ml_model.lower()
        self.sample_rate = sample_rate * 1e6

        # Set features (defaults to ALL_FEATURES if not provided)
        self.features = selected_features if selected_features is not None else self.ALL_FEATURES
        self.feature_indices = [self.ALL_FEATURES.index(f) for f in self.features if f in self.ALL_FEATURES]

        self.hyperparams = hyperparams if hyperparams is not None else {}
        self.model = None
        self.initialize_model()
        self.performance = None

    def initialize_model(self):
        """Initializes model using passed hyperparams (stripping pipeline prefixes if present)."""
        clean_params = {k.replace('model__', ''): v for k, v in self.hyperparams.items()}

        if self.model_type == "random_forest":
            default_params = {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1}
            default_params.update(clean_params)
            self.model = RandomForestClassifier(**default_params)

        elif self.model_type == "xgboost":
            default_params = {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'n_jobs': 1}
            default_params.update(clean_params)
            self.model = xgb.XGBClassifier(**default_params)

        elif self.model_type == "svm":
            default_params = {'kernel': 'rbf', 'class_weight': 'balanced', 'random_state': 42}
            default_params.update(clean_params)
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(**default_params))
            ])

        else:
            raise ValueError(f"Model {self.model_type} is not supported.")

    def _get_scoring_metric(self):
        return 'precision_macro'

    def _filter_features(self, feature_data):
        """Filters feature matrix down to selected feature subset."""
        feature_array = np.array(feature_data)
        # If dataset is already pre-filtered to selected features, return as is
        if feature_array.ndim == 2 and feature_array.shape[1] == len(self.features):
            return feature_array
        return feature_array[:, self.feature_indices]

    def optimize_hyperparams(self, x_train, y_train, param_grid):
        # Subset X using the features already defined in self.features
        feature_indices = [self.ALL_FEATURES.index(f) for f in self.features]
        X = np.array(x_train)[:, feature_indices]
        y_train = np.array(y_train)

        pipeline_steps = []

        # Distance/Gradient based models need scaling
        if self.model_type == "svm":
            pipeline_steps.append(('scaler', StandardScaler()))
            base_estimator = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        else:
            base_estimator = self.model

        pipeline_steps.append(('model', base_estimator))
        pipeline = Pipeline(pipeline_steps)

        scoring_func = self._get_scoring_metric()

        # Use exclusively GridSearchCV
        search_engine = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring=scoring_func,
            n_jobs=-1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)

            if self.model_type == "xgboost":
                sample_weights = compute_sample_weight('balanced', y_train)
                search_engine.fit(X, y_train, model__sample_weight=sample_weights)
            else:
                search_engine.fit(X, y_train)

        self.hyperparams = {
            k.replace('model__', ''): v for k, v in search_engine.best_params_.items() if k.startswith('model__')
        }

        best_score = search_engine.best_score_
        self.best_score = best_score

        self.initialize_model()

        with open(results_hyperparams_filepath, "a", encoding="utf-8") as f:
            f.write(f"=== {self.model_type.upper()} CONFIGURATION & RESULTS ===\n")
            f.write(f"BEST SCORE: {best_score:.4f}\n")
            f.write("HYPERPARAMETERS: ")
            for param, value in self.hyperparams.items():
                f.write(f"{param}: {value}, ")
            f.write(f"\n")
            f.write(f"\n" + f"-" * 30 + f"\n")

        return self

    def evaluate_performance(self, x_train, x_test, y_train, y_test, seconds, filepath=None, board="all", write_to_file=False):

        # 2. Perform the split internally (stratified to maintain class balance)
        X_train, X_test, y_train, y_test = self._filter_features(x_train), self._filter_features(x_test), np.array(y_train), np.array(y_test)

        # 3. Fit the model exclusively on the training split
        self.model.fit(X_train, y_train)

        # 4. Predict exclusively on the validation split
        y_pred = self.model.predict(X_test)

        # 5. Evaluate performance
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        report = classification_report(y_test, y_pred, zero_division=0)

        if write_to_file:
            with open(filepath, "a", encoding="utf-8") as f:
                if board == "car":
                    f.write(f"=== {self.model_type.upper()} PERFORMANCE RESULTS for {seconds} seconds of recording at Sample Rate: {self.sample_rate / 1e6:.1f} MHz for car ===\n")
                else:
                    f.write(f"=== {self.model_type.upper()} PERFORMANCE RESULTS for {seconds} seconds of recording at Sample Rate: {self.sample_rate / 1e6:.1f} MHz for {board} boards ===\n")
                f.write(f"Validation Macro Precision: {macro_prec:.2f}%\n")
                f.write(f"-" * 30 + f"\n")



        self.performance = macro_prec

        return report

    def evaluate_cv_folds(self, x_train, y_train, a_or_w, n_splits=5):
        """Evaluates model performance across CV folds using macro precision and writes results directly to a file."""
        X = self._filter_features(x_train)
        y = np.array(y_train)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring_metric = self._get_scoring_metric()

        fold_scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring_metric, n_jobs=-1)

        mean_prec = np.mean(fold_scores) * 100
        std_prec = np.std(fold_scores) * 100

        lines = [
            f"=== {n_splits}-FOLD CV BREAKDOWN: {self.model_type.upper()} ===",
            f"Sample Rate: {self.sample_rate / 1e6:.1f} MHz\n"
        ]

        for fold_idx, score in enumerate(fold_scores, start=1):
            lines.append(f"  Fold {fold_idx}: {score * 100:.2f}%")

        lines.extend([
            "-" * 40,
            f"  Mean Precision (Macro): {mean_prec:.2f}%",
            f"  Std Deviation         : ±{std_prec:.2f}%"
        ])

        summary_text = "\n".join(lines)
        with open(results_folds_and_training_splits_filepath, a_or_w, encoding="utf-8") as f:
            f.write(summary_text)
            f.write(f"\n" + f"-" * 30 + f"\n")

        return fold_scores

    def evaluate_train_val_splits(self, x_train, y_train, a_or_w, n_iterations=5):
        """
        Tests model performance across varying training sizes (10% to 90%) using Macro Precision.
        Runs multiple iterations per split to ensure stable results, then saves to a file.
        """
        X = self._filter_features(x_train)
        y = np.array(y_train)

        train_percentages = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        lines = [
            f"=== DATA QUANTITY ANALYSIS: {self.model_type.upper()} ===",
            f"Sample Rate: {self.sample_rate / 1e6:.1f} MHz",
            f"Iterations per split: {n_iterations}",
            "-" * 65,
            "Format: Train % / Val % -> Mean Precision (Macro) ± Std Dev",
            "-" * 65
        ]

        for train_pct in train_percentages:
            val_pct = round(1.0 - train_pct, 2)
            sss = StratifiedShuffleSplit(n_splits=n_iterations, train_size=train_pct, test_size=val_pct, random_state=42)

            fold_precisions = []

            for train_index, val_index in sss.split(X, y):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                self.model.fit(X_train, y_train)

                raw_preds = self.model.predict(X_val)

                y_pred = raw_preds

                prec = precision_score(y_val, y_pred, average='macro', zero_division=0) * 100
                fold_precisions.append(prec)

            mean_prec = np.mean(fold_precisions)
            std_prec = np.std(fold_precisions)

            line = (f"Train {int(train_pct * 100):2d}% / Val {int(val_pct * 100):2d}% : "
                    f"{mean_prec:5.2f}% ± {std_prec:4.2f}% ")
            lines.append(line)
            print(line)

        summary_text = "\n".join(lines)

        with open(results_folds_and_training_splits_filepath, a_or_w, encoding="utf-8") as f:
            f.write(summary_text)
            f.write(f"\n" + f"-" * 30 + f"\n")


        return summary_text

    def test_feature_subsets(self, x_train, y_train, results_filepath, max_combinations=150, min_features=5, max_features=25, n_splits=3,
                             inclusion_threshold=65.0):

        # 1. Establish the pool of features
        feature_pool = self.features.copy()
        X_base = self._filter_features(x_train)
        y_train = np.array(y_train)

        actual_max_features = min(max_features, len(feature_pool))
        actual_min_features = min(min_features, actual_max_features)

        allowed_sizes = [5, 8, 12, 16, 20, 25]

        # 2. Generate random subsets
        subsets = set()
        attempts = 0
        max_attempts = max_combinations * 10

        while len(subsets) < max_combinations and attempts < max_attempts:
            k = random.choice(allowed_sizes)
            if k <= len(feature_pool):
                combo = tuple(sorted(random.sample(feature_pool, k)))
                subsets.add(combo)
            attempts += 1

        print(
            f"Pool size: {len(feature_pool)} features. Testing {len(subsets)} random combinations for statistical analysis...")

        # Backup current class state
        original_features = self.features.copy()
        original_indices = self.feature_indices.copy()

        results = []
        scoring_metric = make_scorer(precision_score, average='macro', zero_division=0)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 3. Evaluate each subset
        for idx, combo in enumerate(subsets, 1):
            combo_list = list(combo)
            local_indices = [feature_pool.index(f) for f in combo_list]
            X_subset = X_base[:, local_indices]

            self.features = combo_list
            self.feature_indices = [self.ALL_FEATURES.index(f) for f in combo_list]

            try:
                scores = cross_val_score(self.model, X_subset, y_train, cv=skf, scoring=scoring_metric, n_jobs=-1)
                mean_score = np.mean(scores) * 100
                std_score = np.std(scores) * 100
            except Exception as e:
                print(f"Error evaluating combination {idx}: {e}")
                mean_score = 0.0
                std_score = 0.0

            results.append({
                'features': combo_list,
                'num_features': len(combo_list),
                'mean_score': mean_score,
                'std_score': std_score
            })

            if idx % 25 == 0 or idx == len(subsets):
                print(f"Tested {idx}/{len(subsets)} combinations...")

        # Sort individual results by score descending
        results.sort(key=lambda x: x['mean_score'], reverse=True)

        # 4. Statistical Analysis Groupings
        total_tested = len(results)

        # A. Performance grouped by number of features (Size Statistics)
        size_stats = defaultdict(list)
        for res in results:
            size_stats[res['num_features']].append(res['mean_score'])

        # B. Feature Consensus Analysis (Top 20% tier)
        top_tier_count = max(5, int(total_tested * 0.20))
        top_results = results[:top_tier_count]
        top_features_list = [f for res in top_results for f in res['features']]
        top_counts = Counter(top_features_list)

        feature_stats = []
        for feat in feature_pool:
            appearances_in_top = top_counts.get(feat, 0)
            inclusion_rate = (appearances_in_top / top_tier_count) * 100

            scores_with_feat = [res['mean_score'] for res in results if feat in res['features']]
            avg_score_with = np.mean(scores_with_feat) if scores_with_feat else 0.0

            feature_stats.append({
                'feature': feat,
                'inclusion_rate': inclusion_rate,
                'avg_score_with': avg_score_with
            })

        feature_stats.sort(key=lambda x: x['inclusion_rate'], reverse=True)

        # 5. Extract Robust Feature Set based on Threshold
        robust_features = [stat['feature'] for stat in feature_stats if stat['inclusion_rate'] >= inclusion_threshold]

        if len(robust_features) < actual_min_features:
            robust_features = [stat['feature'] for stat in feature_stats[:actual_min_features]]
        elif len(robust_features) > actual_max_features:
            robust_features = robust_features[:actual_max_features]

        # 6. Evaluate the Robust Feature Set via Cross-Validation
        local_indices = [feature_pool.index(f) for f in robust_features]
        X_robust = X_base[:, local_indices]

        self.features = robust_features
        self.feature_indices = [self.ALL_FEATURES.index(f) for f in robust_features]

        try:
            robust_scores = cross_val_score(self.model, X_robust, y_train, cv=skf, scoring=scoring_metric, n_jobs=-1)
            robust_score = np.mean(robust_scores) * 100
            robust_std = np.std(robust_scores) * 100
        except Exception as e:
            print(f"Error evaluating robust feature set: {e}")
            robust_score = 0.0
            robust_std = 0.0

        # 7. Restore original class state
        self.features = original_features
        self.feature_indices = original_indices

        # 8. Save everything to text file
        with open(results_filepath, "w", encoding="utf-8") as f:
            f.write(f"=== STATISTICAL CONSENSUS & RANDOM SEARCH ANALYSIS: {self.model_type.upper()} ===\n")
            f.write(f"Sample Rate: {self.sample_rate / 1e6:.1f} MHz\n")
            f.write(f"Total Combinations Tested: {total_tested}\n")
            f.write(f"Inclusion Threshold: {inclusion_threshold}% (based on top {top_tier_count} models)\n")
            f.write("-" * 80 + "\n")

            # 1. Robust Result Summary
            f.write(f"ROBUST CONSENSUS FEATURE SET RESULT: {robust_score:.2f}% (±{robust_std:.2f}%)\n")
            f.write(f"Selected Features ({len(robust_features)}): {', '.join(robust_features)}\n")
            f.write("-" * 80 + "\n")

            # 2. Performance Breakdown by Feature Count Table
            f.write("STATISTICAL SUMMARY BY FEATURE COUNT:\n")
            f.write(f"{'Num Features':<15} | {'Average Score':<15} | {'Max Score':<15} | {'Min Score':<15}\n")
            f.write("-" * 65 + "\n")
            for k in sorted(size_stats.keys()):
                scores_for_k = size_stats[k]
                avg_k = np.mean(scores_for_k)
                max_k = np.max(scores_for_k)
                min_k = np.min(scores_for_k)
                f.write(f"{k:<15} | {avg_k:5.2f}%         | {max_k:5.2f}%         | {min_k:5.2f}%\n")
            f.write("-" * 80 + "\n\n")

            # 3. Consensus Table
            f.write(f"FEATURE INCLUSION CONSENSUS:\n")
            f.write(f"{'Feature Name':<25} | {'Top-Tier Inclusion':<20} | {'Avg Score With':<15}\n")
            f.write("-" * 65 + "\n")
            for stat in feature_stats:
                f.write(
                    f"{stat['feature']:<25} | {stat['inclusion_rate']:5.1f}%              | {stat['avg_score_with']:5.2f}%\n")

            f.write("-" * 80 + "\n")

            # 4. Individual Tested Combinations (Ranked)
            f.write(f"ALL TESTED RANDOM COMBINATIONS (Ranked):\n")
            f.write("-" * 80 + "\n")
            for rank, res in enumerate(results, 1):
                f.write(f"Rank {rank}: {res['mean_score']:.2f}% (±{res['std_score']:.2f}%), ")
                f.write(f"Number of Features: {res['num_features']}\n")
                f.write(f"Features: {', '.join(res['features'])}\n")
                f.write("-" * 80 + "\n")

            f.write("\n" + "=" * 80 + "\n\n")

        print(f"Analysis complete. Full statistical report saved to '{results_filepath}'")

        return robust_score, robust_features, top_results

    def fit(self, feature_data, y=None):
        filtered_data = self._filter_features(feature_data)
        self.model.fit(filtered_data, y)
        return self

    def predict(self, feature_data, y_ground_truth=None):
        filtered_data = self._filter_features(feature_data)
        raw_preds = self.model.predict(filtered_data)
        return raw_preds

