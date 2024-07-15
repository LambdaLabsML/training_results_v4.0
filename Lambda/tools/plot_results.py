import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extract_gpus(folder_name):
    match = re.search(r'(\d+)x(\d+)x(\d+)x(\d+)', folder_name)
    if match:
        a, b, _, _ = map(int, match.groups())
        return a * b
    return None

def plot_metric(df, s_std, metric_mean, metric_std, metric_name, output_path):
    df['GPUs'] = df['folder'].apply(extract_gpus)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['GPUs'], df[metric_mean], yerr=df[metric_std] * s_std, fmt='o-', capsize=5, label=metric_name)
    
    for x, y, yerr in zip(df['GPUs'], df[metric_mean], df[metric_std]):
        plt.text(x, y, f'{y:.2f} ± {yerr*s_std:.2f}', fontsize=8, ha='right')
    
    plt.xlabel('Number of GPUs')
    plt.ylabel(f'{metric_name} Mean, {s_std} Standard Deviation')
    plt.title(f'{metric_name} vs Number of GPUs')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_path, f'{metric_name}.png'))
    plt.close()


if __name__ == "__main__":
    csv_file = "../benchmarks/bert/implementations/pytorch/results/output.csv"
    output_path = "../benchmarks/bert/implementations/pytorch/results"
    s_std = 2
    df = pd.read_csv(csv_file)
    
    plot_metric(df, s_std, 'e2e_time_mean', 'e2e_time_std', 'e2e_time', output_path)
    plot_metric(df, s_std, 'raw_train_time_mean', 'raw_train_time_std', 'raw_train_time', output_path)
    plot_metric(df, s_std, 'training_sequences_per_second_mean', 'training_sequences_per_second_std', 'training_sequences_per_second', output_path)
