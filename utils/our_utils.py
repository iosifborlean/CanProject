import csv
import json
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd


from utils.feature_extraction import *
from utils.ml_model import MLmodel

sample_rate = 50
max_combinations=150

caches_directory = Path(__file__).resolve().parents[1] / "caches"
results_directory = Path(__file__).resolve().parents[1] / "results"

id_dictionary_prototype = {
'301': 1, '328': 1, '338': 2, '350': 1, '35C': 1,
'230': 2, '30E': 1, '33A': 1,
'3D3': 1, '3D4': 1, '3D5': 1, '3D6': 1,
'24F': 50, '36A': 1,
'326': 1,
'12C': 10, '220': 100, '221': 50,
'310': 1, '340': 2, '372': 1, '3C8': 10,
'260': 2, '2CF': 2, '339': 2,
'184': 10, '1C0': 5, '39C': 1,
'2FA': 50,
'102': 100, '23A': 50, '318': 10,
'304': 1,
'126': 100, '1D0': 1, '1E0': 1, '21E': 10,
'144': 50, '170': 50, '178': 10, '1A0': 50, '2F9': 100,
'2A6': 25, '2A8': 25, '2AC': 25, '2C2': 25, '2C6': 25, '3C2': 1
}
id_dictionary_car = {
    '108': 100, '10C': 50, '10D': 50, '118': 100, '11C': 50,
    '124': 50, '126': 100, '12C': 10, '134': 50, '140': 50,
    '144': 50, '14B': 50, '14C': 50, '158': 50, '170': 50,
    '178': 10, '180': 1, '1D0': 1, '1E0': 1, '212': 10,
    '221': 50, '22F': 50, '24F': 50, '278': 10, '2BF': 10,
    '2CF': 2, '2F9': 100, '304': 1, '30B': 1,
    '30C': 10, '32F': 10, '334': 1, '3B4': 0.5
}

param_grid_rf = {
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 10]

}
param_grid_xgb = {
    'model__max_depth': [3, 6],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8, 1.0]
}
param_grid_svm = {
    'model__kernel': ['rbf', 'linear', 'poly'],
    'model__C': [0.1, 1.0, 10.0],
    'model__gamma': ['scale', 0.01, 0.1]
}

grids = {
    "random_forest": param_grid_rf,
    "xgboost": param_grid_xgb,
    "svm": param_grid_svm,
}

combinazioni_board = {"aurix": ['301', '328', '338', '350', '35C', '230', '30E', '33A', '3D3', '3D4', '3D5', '3D6', '24F', '36A', '326'],
                   "st": ['12C', '220', '221', '310', '340', '372', '3C8', '260', '2CF', '339', '184', '1C0', '39C', '2FA'],
                   "atmel": ['102', '23A', '318', '304', '126', '1D0', '1E0', '21E', '144', '170', '178', '1A0', '2F9', '2A6', '2A8', '2AC', '2C2', '2C6', '3C2'],
                    "all": ['301', '328', '338', '350', '35C', '230', '30E', '33A', '3D3', '3D4', '3D5', '3D6', '24F', '36A', '326',
                          '12C', '220', '221', '310', '340', '372', '3C8', '260', '2CF', '339', '184', '1C0', '39C', '2FA',
                          '102', '23A', '318', '304', '126', '1D0', '1E0', '21E', '144', '170', '178', '1A0', '2F9', '2A6', '2A8', '2AC', '2C2', '2C6', '3C2']
                   }

BITS_PER_FRAME = 32

def parse_file(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for _ in range(1):
            next(file)
        csvFile = csv.reader(file)
        for line in csvFile:
            lines.append(line)
    return lines

def get_all_voltages(lst):
   return [float(voltages[1]) - float(voltages[2]) for voltages in lst]

def find_ecu(matrix, element):
    for i, sublist in enumerate(matrix):
        if element in sublist:
            return i
    return -1

def load_and_split_voltage_dataset(downsampling, car_type, target_ids, recording_time):
    dataset_path = Path(__file__).resolve().parents[1] / f'voltages/{car_type}'
    features_dataset = []
    file_names = os.listdir(dataset_path)

    if car_type == "prototype":
        id_dictionary = id_dictionary_prototype
    else:
        id_dictionary = id_dictionary_car

    for file_name in file_names:
        file = os.path.join(dataset_path, file_name)
        current_id = file_name.split('_')[0]
        if current_id not in target_ids:
            continue

        current_count = int(file_name.split('_')[1].split('.')[0])
        id_frequency = id_dictionary[current_id]
        max_samples = id_frequency * recording_time * BITS_PER_FRAME
        if current_count >= max_samples:
            continue

        lines = parse_file(file)
        differential = get_all_voltages(lines)
        differential = np.array(differential)

        step = int(downsampling)
        if step > 1:
            differential = differential[::step]

        feature = [current_id, differential]
        features_dataset.append(feature)

    clustering_save_path = Path(__file__).resolve().parents[1] / f"results/clusters_{car_type}.csv"
    ecu_mapping = []

    with open(clustering_save_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and re.match(r'^\d', row[0]):
                ecu_mapping.append(row[1])
    ecu_mapping = [eval(item) for item in ecu_mapping]

    id_list = [row[0] for row in features_dataset]
    unique_id_list = list(dict.fromkeys(id_list))

    group_dataset = [[] for _ in range(len(unique_id_list))]
    for row in features_dataset:
        pos = unique_id_list.index(row[0])
        group_dataset[pos].append(row)

    split_ratio = 0.8
    train_dataset, test_dataset = [], []
    for i in range(len(unique_id_list)):
        sub_dataset = group_dataset[i]
        split_idx = int(len(sub_dataset) * split_ratio)
        train_dataset.append(sub_dataset[:split_idx])
        test_dataset.append(sub_dataset[split_idx:])

    merged_train_dataset_list = [item for sublist in train_dataset for item in sublist]
    merged_test_dataset_list = [item for sublist in test_dataset for item in sublist]

    random.shuffle(merged_train_dataset_list)
    random.shuffle(merged_test_dataset_list)

    X_train = [[row[0], row[1]] for row in merged_train_dataset_list]
    X_test = [[row[0], row[1]] for row in merged_test_dataset_list]

    merged_train_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_train_dataset_list]
    merged_test_dataset_y_numerical = [find_ecu(ecu_mapping, row[0]) for row in merged_test_dataset_list]

    Y_train = np.array(merged_train_dataset_y_numerical).astype(float)
    Y_test = np.array(merged_test_dataset_y_numerical).astype(float)

    return X_train, X_test, Y_train, Y_test

def feature_extract(sample, sample_rate):
        data = sample[1]
        data_len = len(data)
        sigma = 1.0 / sample_rate
        target_window_time = 300 * 1e-9
        fixed_range = int(np.round(target_window_time / sigma))

        # 1. Safe Index Clamping
        mid_index = int(np.floor(data_len / 2))
        start_index = max(0, mid_index - fixed_range)
        end_index = min(data_len - 1, mid_index + fixed_range)

        # Base calculations for domain features
        v_mean = np.mean(data[start_index:end_index + 1])
        v_max = np.max(data[:start_index + 1])
        min_left = np.min(data[:mid_index])
        min_right = np.min(data[mid_index:])

        signal_range = v_max - min(min_left, min_right)
        base_epsilon = max(0.02, 0.05 * signal_range) if signal_range > 0 else 0.02

        # 2. Extract Timing Features
        t_bit = calc_t_bit(data, mid_index, min_left, min_right, base_epsilon, sigma, data_len)
        t_plat, dominant_start, dominant_end = calc_t_plat(data, v_mean, mid_index, base_epsilon, sigma, start_index,
                                                           end_index)

        # Slice dominant data
        dom_data = data[dominant_start:dominant_end + 1]
        if len(dom_data) == 0:
            dom_data = data

        # ==========================================
        # 3. TIME-DOMAIN FEATURES
        # ==========================================
        time_maximum = np.max(dom_data)
        time_minimum = np.min(dom_data)
        time_mean = np.mean(dom_data)
        peak = np.max(np.abs(dom_data))
        mean_absolute = np.mean(np.abs(dom_data))
        variance = np.var(dom_data)
        std_dev = np.std(dom_data)
        rms = calc_rms(dom_data)

        skewness = calc_time_skewness(dom_data, time_mean, std_dev)
        time_kurtosis = calc_time_kurtosis(dom_data, time_mean, std_dev)
        form_factor = calc_form_factor(dom_data, rms, mean_absolute)
        peak_factor = calc_peak_factor(dom_data, peak, rms)
        impulse_factor = calc_impulse_factor(dom_data, peak, mean_absolute)
        crest_factor = calc_crest_factor(dom_data, peak)

        # ==========================================
        # 4. FREQUENCY-DOMAIN FEATURES
        # ==========================================
        s = np.abs(np.fft.rfft(dom_data))
        N_freq = len(s)
        f_k = np.arange(1, N_freq + 1)
        sum_s = np.sum(s) + 1e-12

        freq_peak = np.max(np.abs(s))
        freq_mean = np.mean(s)

        centroid = calc_freq_centroid(s, f_k, sum_s)
        arithmetic_mean = calc_freq_arithmetic_mean(s, N_freq)
        freq_crest = calc_freq_crest(s, arithmetic_mean)
        decrease = calc_freq_decrease(s, N_freq)
        entropy = calc_freq_entropy(s, N_freq)
        geometric_mean = calc_freq_geometric_mean(s, N_freq)
        flatness = calc_freq_flatness(s, geometric_mean, arithmetic_mean)
        flux = calc_freq_flux(s)

        spread = calc_freq_spread(s, f_k, centroid, sum_s)
        freq_skewness = calc_freq_skewness(s, f_k, centroid, spread, sum_s)
        freq_kurtosis = calc_freq_kurtosis(s, f_k, centroid, spread, sum_s)

        rolloff_point = calc_freq_rolloff(s, f_k)
        slope = calc_freq_slope(s, f_k, freq_mean)

        return (v_mean, v_max, t_bit, t_plat,
                time_maximum, time_minimum, time_mean, peak, mean_absolute, variance,
                std_dev, time_kurtosis, skewness, rms, form_factor, peak_factor,
                impulse_factor, crest_factor,
                centroid, freq_crest, freq_peak, freq_mean, decrease, entropy,
                flatness, arithmetic_mean, geometric_mean, flux, freq_kurtosis,
                rolloff_point, freq_skewness, slope, spread)

def load_or_extract_features(sample_rate, recording_time, config, target_ids, car_type, downsampling):
    # 1. Load or Create Raw Data Cache
    cache_raw_data_path = caches_directory / f"{car_type}_raw_{config}_{sample_rate}Ms_{recording_time}s"

    if cache_raw_data_path.exists():
        cached_data = joblib.load(cache_raw_data_path)
        X_train = cached_data['X_train']
        X_test = cached_data['X_test']
        Y_train = cached_data['Y_train']
        Y_test = cached_data['Y_test']
    else:
        X_train, X_test, Y_train, Y_test = load_and_split_voltage_dataset(downsampling, car_type, target_ids,
                                                                          recording_time)
        cache_payload = {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test
        }
        joblib.dump(cache_payload, cache_raw_data_path)

    # 2. Load or Extract Train Feature Data
    cache_feature_train_data_path = caches_directory / f"{car_type}_training_features_{config}_{sample_rate}Ms_{recording_time}s"

    if cache_feature_train_data_path.exists():
        with open(cache_feature_train_data_path, 'rb') as f:
            feature_data_train = pickle.load(f)
    else:
        feature_data_train = [feature_extract(sample, sample_rate) for sample in X_train]

        with open(cache_feature_train_data_path, 'wb') as f:
            pickle.dump(feature_data_train, f)

    cache_feature_test_data_path = caches_directory / f"{car_type}_testing_features_{config}_{sample_rate}Ms_{recording_time}s"
    if cache_feature_test_data_path.exists():
        with open(cache_feature_test_data_path, 'rb') as f:
            feature_data_test = pickle.load(f)
    else:
        feature_data_test = [feature_extract(sample, sample_rate) for sample in X_test]

        with open(cache_feature_test_data_path, 'wb') as f:
            pickle.dump(feature_data_test, f)

    return feature_data_train, feature_data_test, Y_train, Y_test #which are properly x_train, x_test, y_train, y_test

def best_model():
    file_path = Path(__file__).resolve().parents[1] / "results/best_model"

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]
            hyperparams = json.loads(row["hyperparams"])  # Converts JSON string to dict
            features = json.loads(row["features"])
    return model_name, hyperparams, features

def model_optimization_and_choice():
    downsampling = 1
    car_type = 'prototype'
    model_types = ["xgboost", "random_forest", "svm"]
    config, target_ids = "all", combinazioni_board["all"]
    recording_time = 35
    models_performances = []

    real_sample_rate = 50 / downsampling

    results_filepath = results_directory / "best_hyperparams.txt"
    open(results_filepath, "w").close()

    x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config, target_ids, car_type, downsampling)
    for model_type in model_types:
        opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type)

        opt_model.optimize_hyperparams(
            x_train=x_train,
            y_train=y_train,
            param_grid=grids[model_type],
        )

        opt_model.evaluate_performance(x_train, x_test, y_train, y_test, recording_time)

        models_performances.append(opt_model)

        print(f"hyperparams optimizization done for {model_type}")

    best_model = max(models_performances, key=lambda m: m.performance)

    # 2. Extract data ONLY for the best model
    best_model_data = [{
        'model_name': best_model.model_type,
        'hyperparams': json.dumps(best_model.hyperparams),
        'features': json.dumps(best_model.features),
    }]

    # 3. Export to CSV
    df_best = pd.DataFrame(best_model_data)
    df_best.to_csv(results_directory / "best_model", index=False)

def evaluate_folds_and_splits():
    model_type, hyperparams, features = best_model()

    downsampling = 1
    car_type = 'prototype'
    config, target_ids = "all", combinazioni_board["all"]
    recording_time = 35

    real_sample_rate = 50 / downsampling

    x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config, target_ids,
                                                                car_type, downsampling)

    opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type, selected_features=features, hyperparams=hyperparams)

    opt_model.evaluate_cv_folds(x_train=x_train, y_train=y_train, a_or_w="w")
    opt_model.evaluate_train_val_splits(x_train=x_train, y_train=y_train, a_or_w="a")


def evaluate_recording_times():
    while True:
        car_type = str(input("\nChoose car type ('prototype' or 'car'): "))
        if car_type == 'prototype':
            config, target_ids = "all", combinazioni_board["all"]
            results_filepath = results_directory / "recording_times_prototype.txt"
            report_file = results_directory / "reports_prototype"
            open(results_filepath, "w").close()
            break
        elif car_type == 'car':
            config, target_ids = "car", id_dictionary_car.keys()
            results_filepath = results_directory / "recording_times_car.txt"
            report_file = results_directory / "reports_car"
            open(results_filepath, "w").close()
            break
        else:
            print("Options are 'prototype' and 'car'")


    model_type, hyperparams, features = best_model()
    downsampling = 1

    real_sample_rate = 50 / downsampling

    recording_times = [1, 2, 3, 5, 7, 10, 15, 20, 30]

    with open(report_file, "w", encoding="utf-8") as f:
        pass

    for recording_time in recording_times:
        x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config, target_ids,
                                                                    car_type, downsampling)

        opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type, selected_features=features,
                            hyperparams=hyperparams)

        report = opt_model.evaluate_performance(x_train, x_test, y_train, y_test, recording_time, board=car_type, filepath=results_filepath, write_to_file=True)
        with open(report_file, "a", encoding="utf-8") as f:
            if car_type == "car":
                f.write(
                    f"=== {model_type} PERFORMANCE RESULTS for {recording_time} seconds of recording at Sample Rate: {real_sample_rate} MHz for car ===\n")
            else:
                f.write(
                    f"=== {model_type} PERFORMANCE RESULTS for {recording_time} seconds of recording at Sample Rate: {real_sample_rate} MHz for {model_type} boards ===\n")
            f.write(report)
            f.write(f"-" * 30 + f"\n")

def evaluate_single_board_type():
    model_type, hyperparams, features = best_model()

    downsampling = 1
    car_type = 'prototype'
    recording_time = 35
    results_filepath = results_directory / "single_board_results.txt"

    real_sample_rate = 50 / downsampling

    for config, target_ids in combinazioni_board.items():
        x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config,
                                                                    target_ids,
                                                                    car_type, downsampling)
        if config == "st":
            y_train = y_train.astype(int) - 5
            y_test = y_test.astype(int) - 5
        if config == "atmel":
            y_train = y_train.astype(int) - 10
            y_test = y_test.astype(int) - 10

        opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type, selected_features=features,
                            hyperparams=hyperparams)

        opt_model.evaluate_performance(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seconds=recording_time, board=config, filepath=results_filepath, write_to_file=True)

def evaluate_downsampling_performance():
    model_type, hyperparams, features = best_model()

    downsamplings = [1, 2, 3, 4, 5, 6, 7, 8]
    car_type = 'prototype'
    config, target_ids = "all", combinazioni_board["all"]
    recording_time = 35

    results_filepath = results_directory / "downsampling_results.txt"

    for downsampling in downsamplings:
        real_sample_rate = 50 / downsampling

        x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config, target_ids,
                                                                    car_type, downsampling)

        opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type, selected_features=features,
                            hyperparams=hyperparams)

        opt_model.evaluate_performance(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seconds=recording_time, board=config, filepath=results_filepath, write_to_file=True)

def evaluate_features():
    model_type, hyperparams, features = best_model()

    downsampling = 1
    car_type = 'prototype'
    config, target_ids = "all", combinazioni_board["all"]
    recording_time = 35
    real_sample_rate = 50 / downsampling

    results_filepath = results_directory / "features_results.txt"

    x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config, target_ids,
                                                                car_type, downsampling)

    opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type, selected_features=features,
                        hyperparams=hyperparams)

    robust_score, robust_features, top_results = opt_model.test_feature_subsets(x_train=x_train, y_train=y_train, max_combinations=max_combinations, results_filepath=results_filepath)

    best_model_csv_path = results_directory / "best_model.csv"
    if best_model_csv_path.exists():
        df = pd.read_csv(best_model_csv_path)
        df.loc[df['model_name'] == model_type, 'features'] = json.dumps(robust_features)
        df.to_csv(best_model_csv_path, index=False)
        print(f"Rewrote updated robust features to {best_model_csv_path}")


def evaluate_downsampling_and_features_performance():
    model_type, hyperparams, features = best_model()

    global_top_feature_counter = Counter()

    downsamplings = [1, 2, 3, 4, 8]
    car_type = 'prototype'
    config, target_ids = "all", combinazioni_board["all"]
    recording_time = 35
    results_filepath = results_directory / "downsampling_and_features_results.txt"

    for downsampling in downsamplings:
        real_sample_rate = 50 / downsampling

        x_train, x_test, y_train, y_test = load_or_extract_features(real_sample_rate, recording_time, config,
                                                                    target_ids, car_type, downsampling)
        opt_model = MLmodel(sample_rate=real_sample_rate, ml_model=model_type,
                            hyperparams=hyperparams)

        score, features, top_results = opt_model.test_feature_subsets(x_train=x_train, y_train=y_train, max_combinations=max_combinations, results_filepath=results_filepath)

        for res in top_results:
            for feat in res['features']:
                global_top_feature_counter[feat] += 1

    total_top_models_globally = len(downsamplings) * (max_combinations*0.2)

    global_stats = []
    for feat, count in global_top_feature_counter.items():
        global_inclusion_rate = (count / total_top_models_globally) * 100
        global_stats.append(
            {'feature': feat, 'global_inclusion': global_inclusion_rate, 'total_appearances': count})

    global_stats.sort(key=lambda x: x['global_inclusion'], reverse=True)

    # Save the Global Summary to a master file
    with open(results_filepath, "w", encoding="utf-8") as f:
        f.write(f"=== GLOBAL CROSS-SAMPLING-RATE FEATURE CONSENSUS ===\n")
        f.write(f"Sampling Rates Tested: {[round(50 / x, 2) for x in downsamplings]} MHz\n")
        f.write(f"Total Top-Tier Model Instances Evaluated: {total_top_models_globally}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Feature Name':<25} | {'Global Inclusion Rate':<22} | {'Total Appearances':<15}\n")
        f.write("-" * 65 + "\n")

        for stat in global_stats:
            f.write(
                f"{stat['feature']:<25} | {stat['global_inclusion']:5.1f}%                 | {stat['total_appearances']}/{total_top_models_globally}\n")

        f.write("-" * 65 + "\n")

    print(f"\nGlobal cross-sampling analysis complete! Saved to '{results_filepath}'")