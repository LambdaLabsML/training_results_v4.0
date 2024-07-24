import os
import re
import datetime
import pandas as pd
import json
from statistics import mean, stdev


def llama2_70b_raw_training_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    time_ms_samples_count_0 = None
    time_ms_status_success = None

    for line in lines:
        if "time_ms" in line:
            # Extract the JSON part of the line
            json_part = line.split(':::MLLOG ')[-1]
            line_data = json.loads(json_part)
            if "samples_count" in json_part and line_data["metadata"]["samples_count"] == 0:
                time_ms_samples_count_0 = line_data["time_ms"]
            if "\"status\"" in json_part and line_data["metadata"]["status"] == "success":
                time_ms_status_success = line_data["time_ms"]

            if time_ms_samples_count_0 is not None and time_ms_status_success is not None:
                break

    if time_ms_samples_count_0 is None or time_ms_status_success is None:
        return -1

    # Convert milliseconds to minutes and compute the difference
    time_difference = (time_ms_status_success - time_ms_samples_count_0) / 60000.0

    return time_difference


def llama2_70b_e2e_time(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Get the last two lines
    last_line = lines[-1].strip()
    second_last_line = lines[-2].strip()

    # Extract the time strings from the lines
    start_time_str = last_line.split(' ')[-3].split(',')[-1] + ' ' + last_line.split(' ')[-2] + ' ' + last_line.split(' ')[-1]
    end_time_str = second_last_line.split(' ')[-3] + ' ' + second_last_line.split(' ')[-2] + ' ' + second_last_line.split(' ')[-1]

    # Convert the time strings to datetime objects
    try:
        start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %I:%M:%S %p')
        end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %I:%M:%S %p')
    except ValueError:
        return -1
    
    # Compute the difference in minutes
    time_difference = (end_time - start_time).total_seconds() / 60.0
    return time_difference


def llama2_70b_throughput(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    training_sequences_per_second = re.findall(r"\"throughput\": (\d+\.\d+)", content) 
    return training_sequences_per_second

def extract_metrics(log_file, name):
    if name == "bert":
        with open(log_file, 'r') as file:
             content = file.read()
        e2e_time = re.findall(r"'e2e_time': (\d+\.\d+)", content)
        training_sequences_per_second = re.findall(r"'training_sequences_per_second': (\d+\.\d+)", content)
        raw_train_time = re.findall(r"'raw_train_time': (\d+\.\d+)", content)
    elif name == "llama2-70b":
        e2e_time = [llama2_70b_e2e_time(log_file)]      
        training_sequences_per_second = llama2_70b_throughput(log_file)
        raw_train_time = [llama2_70b_raw_training_time(log_file)]
    else:
        e2e_time = [0]
        training_sequences_per_second = [0]
        raw_train_time = [0]

    e2e_time = [float(x) for x in e2e_time]
    training_sequences_per_second = [float(x) for x in training_sequences_per_second]
    raw_train_time = [float(x) for x in raw_train_time]

    return e2e_time, training_sequences_per_second, raw_train_time

def compute_mean_std(metrics):
    if len(metrics) > 1:
        return (mean(metrics), stdev(metrics))
    else:
        return metrics[0], 0.1

def process_folder(folder, name):
    e2e_times = []
    training_sequences_per_seconds = []
    raw_train_times = []

    for log_file in os.listdir(folder):
        if log_file.endswith('.log') and 'nccl' not in log_file:
            e2e_time, training_sequences_per_second, raw_train_time = extract_metrics(os.path.join(folder, log_file), name)
            e2e_times.extend(e2e_time)
            training_sequences_per_seconds.extend(training_sequences_per_second)
            raw_train_times.extend(raw_train_time)

    if not e2e_times or not training_sequences_per_seconds or not raw_train_times:
        return None

    e2e_times = [x for x in e2e_times if x > 0]
    training_sequences_per_seconds = [x for x in training_sequences_per_seconds if x > 0]
    raw_train_times = [x for x in raw_train_times if x > 0]

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

def process_folders(root_folder, name):
    results = []

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            result = process_folder(folder_path, name)
            if result:
                results.append(result)

    return results

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    #name = "bert"
    #root_folder = "../benchmarks/bert/implementations/pytorch/results"
    #output_file = "../benchmarks/bert/implementations/pytorch/results/output.csv"


    name = "llama2-70b"
    root_folder = "../benchmarks/llama2_70b_lora/implementations/nemo/results"
    output_file = "../benchmarks/llama2_70b_lora/implementations/nemo/results/output.csv"

    data = process_folders(root_folder, name)
    save_to_csv(data, output_file)

