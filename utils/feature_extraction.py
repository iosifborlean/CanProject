import numpy as np


# ==========================================
# DOMAIN-SPECIFIC FEATURES (Timing & Voltage)
# ==========================================

def calc_t_bit(data, mid_index, min_left, min_right, base_epsilon, sigma, data_len):
    epsilon_bit = base_epsilon
    while True:
        t_bit_alpha_index = np.where(np.abs(data - min_left) <= epsilon_bit)
        t_bit_beta_index = np.where(np.abs(data - min_right) <= epsilon_bit)

        t_bit_alpha = t_bit_alpha_index[0][t_bit_alpha_index[0] <= mid_index]
        t_bit_beta = t_bit_beta_index[0][t_bit_beta_index[0] > mid_index]

        if len(t_bit_alpha) > 0 and len(t_bit_beta) > 0:
            break

        epsilon_bit *= 1.5
        if epsilon_bit > 2.0:
            t_bit_alpha = np.array([0])
            t_bit_beta = np.array([data_len - 1])
            break

    return (t_bit_beta[0] - t_bit_alpha[-1]) * sigma


def calc_t_plat(data, v_mean, mid_index, base_epsilon, sigma, start_index, end_index):
    epsilon_plat = base_epsilon
    while True:
        t_plat_alpha_beta_index = np.where(np.abs(data - v_mean) <= epsilon_plat)
        t_plat_alpha = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] <= mid_index]
        t_plat_beta = t_plat_alpha_beta_index[0][t_plat_alpha_beta_index[0] > mid_index]

        if len(t_plat_alpha) > 0 and len(t_plat_beta) > 0:
            break

        epsilon_plat *= 1.5
        if epsilon_plat > 2.0:
            t_plat_alpha = np.array([start_index])
            t_plat_beta = np.array([end_index])
            break

    t_plat = (t_plat_beta[-1] - t_plat_alpha[0]) * sigma
    return t_plat, t_plat_alpha[0], t_plat_beta[-1]


# ==========================================
# TIME-DOMAIN FEATURES
# ==========================================

def calc_time_skewness(data, mean=None, std_dev=None):
    if mean is None: mean = np.mean(data)
    if std_dev is None: std_dev = np.std(data)
    if std_dev > 1e-12:
        return np.mean(((data - mean) / std_dev) ** 3)
    return 0.0


def calc_time_kurtosis(data, mean=None, std_dev=None):
    if mean is None: mean = np.mean(data)
    if std_dev is None: std_dev = np.std(data)
    if std_dev > 1e-12:
        return np.mean(((data - mean) / std_dev) ** 4) - 3
    return 0.0


def calc_rms(data):
    return np.sqrt(np.mean(data ** 2))


def calc_form_factor(data, rms=None, mean_absolute=None):
    if rms is None: rms = calc_rms(data)
    if mean_absolute is None: mean_absolute = np.mean(np.abs(data))
    return rms / (mean_absolute + 1e-12)


def calc_peak_factor(data, peak=None, rms=None):
    if peak is None: peak = np.max(np.abs(data))
    if rms is None: rms = calc_rms(data)
    return peak / (rms + 1e-12)


def calc_impulse_factor(data, peak=None, mean_absolute=None):
    if peak is None: peak = np.max(np.abs(data))
    if mean_absolute is None: mean_absolute = np.mean(np.abs(data))
    return peak / (mean_absolute + 1e-12)


def calc_crest_factor(data, peak=None):
    if peak is None: peak = np.max(np.abs(data))
    x_r = (np.mean(np.sqrt(np.abs(data)))) ** 2
    return peak / (x_r + 1e-12)


# ==========================================
# FREQUENCY-DOMAIN FEATURES
# ==========================================
# Note: These take the FFT magnitudes `s` and frequencies `f_k` as input.

def calc_freq_centroid(s, f_k, sum_s=None):
    if sum_s is None: sum_s = np.sum(s) + 1e-12
    return np.sum(f_k * s) / sum_s


def calc_freq_arithmetic_mean(s, N_freq=None):
    if N_freq is None: N_freq = len(s)
    return np.sum(s) / (N_freq - 1 + 1e-12)


def calc_freq_crest(s, arithmetic_mean=None):
    if arithmetic_mean is None: arithmetic_mean = calc_freq_arithmetic_mean(s)
    return np.max(s) / (arithmetic_mean + 1e-12)


def calc_freq_decrease(s, N_freq=None):
    if N_freq is None: N_freq = len(s)
    k_indices = np.arange(1, N_freq)
    if len(k_indices) > 0:
        decrease_num = np.sum((s[1:] - s[0]) / k_indices)
        decrease_den = np.sum(s[1:]) + 1e-12
        return decrease_num / decrease_den
    return 0.0


def calc_freq_entropy(s, N_freq=None):
    if N_freq is None: N_freq = len(s)
    s_safe = s + 1e-12
    return -np.sum(s_safe * np.log(s_safe)) / np.log(N_freq - 1 + 1e-12)


def calc_freq_geometric_mean(s, N_freq=None):
    if N_freq is None: N_freq = len(s)
    s_safe = s + 1e-12
    return np.exp(np.sum(np.log(s_safe)) / (N_freq - 1 + 1e-12))


def calc_freq_flatness(s, geometric_mean=None, arithmetic_mean=None):
    if geometric_mean is None: geometric_mean = calc_freq_geometric_mean(s)
    if arithmetic_mean is None: arithmetic_mean = calc_freq_arithmetic_mean(s)
    return geometric_mean / (arithmetic_mean + 1e-12)


def calc_freq_flux(s):
    return np.sqrt(np.sum(np.abs(np.diff(s)) ** 2))


def calc_freq_spread(s, f_k, centroid=None, sum_s=None):
    if sum_s is None: sum_s = np.sum(s) + 1e-12
    if centroid is None: centroid = calc_freq_centroid(s, f_k, sum_s)
    return np.sqrt(np.sum(((f_k - centroid) ** 2) * s) / sum_s)


def calc_freq_skewness(s, f_k, centroid=None, spread=None, sum_s=None):
    if sum_s is None: sum_s = np.sum(s) + 1e-12
    if centroid is None: centroid = calc_freq_centroid(s, f_k, sum_s)
    if spread is None: spread = calc_freq_spread(s, f_k, centroid, sum_s)

    if spread > 1e-12:
        return np.sum(((f_k - centroid) ** 3) * s) / ((spread ** 3) * sum_s)
    return 0.0


def calc_freq_kurtosis(s, f_k, centroid=None, spread=None, sum_s=None):
    if sum_s is None: sum_s = np.sum(s) + 1e-12
    if centroid is None: centroid = calc_freq_centroid(s, f_k, sum_s)
    if spread is None: spread = calc_freq_spread(s, f_k, centroid, sum_s)

    if spread > 1e-12:
        return np.sum(((f_k - centroid) ** 4) * s) / ((spread ** 4) * sum_s) - 3
    return 0.0


def calc_freq_rolloff(s, f_k, kappa=0.85):
    cumulative_sum = np.cumsum(s)
    threshold = kappa * cumulative_sum[-1]
    rolloff_idx = np.where(cumulative_sum >= threshold)[0]
    return f_k[rolloff_idx[0]] if len(rolloff_idx) > 0 else f_k[-1]


def calc_freq_slope(s, f_k, freq_mean=None):
    if freq_mean is None: freq_mean = np.mean(s)
    mu_f = np.mean(f_k)
    slope_num = np.sum((f_k - mu_f) * (s - freq_mean))
    slope_den = np.sum((f_k - mu_f) ** 2) + 1e-12
    return slope_num / slope_den