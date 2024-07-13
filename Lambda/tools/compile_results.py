import os
import re
import pandas as pd
from statistics import mean, stdev

def extract_metrics(log_file):
    with open(log_file, 'r') as file:
        content = file.read()

    e2e_time = re.findall(r"'e2e_time': (\d+\.\d+)", content)
    training_sequences_per_second = re.findall(r"'training_sequences_per_second': (\d+\.\d+)", content)
    raw_train_time = re.findall(r"'raw_train_time': (\d+\.\d+)", content)

    e2e_time = [float(x) for x in e2e_time]
    training_sequences_per_second = [float(x) for x in training_sequences_per_second]
    raw_train_time = [float(x) for x in raw_train_time]

    return e2e_time, training_sequences_per_second, raw_train_time

def compute_mean_std(metrics):
    return (mean(metrics), stdev(metrics))

def process_folder(folder):
    e2e_times = []
    training_sequences_per_seconds = []
    raw_train_times = []

    for log_file in os.listdir(folder):
        if log_file.endswith('.log'):
            e2e_time, training_sequences_per_second, raw_train_time = extract_metrics(os.path.join(folder, log_file))
            e2e_times.extend(e2e_time)
            training_sequences_per_seconds.extend(training_sequences_per_second)
            raw_train_times.extend(raw_train_time)

    if not e2e_times or not training_sequences_per_seconds or not raw_train_times:
        return None

    e2e_time_mean, e2e_time_std = compute_mean_std(e2e_times)
    training_sequences_per_second_mean, training_sequences_per_second_std = compute_mean_std(training_sequences_per_seconds)
    raw_train_time_mean, raw_train_time_std = compute_mean_std(raw_train_times)

    return {
        'folder': os.path.basename(folder),
        'e2e_time_mean': round(e2e_time_mean, 2),
        'e2e_time_std': round(e2e_time_std, 2),
        'raw_train_time_mean': round(raw_train_time_mean, 2),
        'raw_train_time_std': round(raw_train_time_std, 2),
        'training_sequences_per_second_mean': round(training_sequences_per_second_mean, 2),
        'training_sequences_per_second_std': round(training_sequences_per_second_std, 2),
    }

def process_folders(root_folder):
    results = []

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            result = process_folder(folder_path)
            if result:
                results.append(result)

    return results

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    root_folder = "../benchmarks/bert/implementations/pytorch/results"
    output_file = "../benchmarks/bert/implementations/pytorch/results/output.csv"
    data = process_folders(root_folder)
    save_to_csv(data, output_file)
