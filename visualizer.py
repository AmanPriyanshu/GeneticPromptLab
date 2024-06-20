import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def visualize_performance(run_id):
    run_path = os.path.join('runs', run_id)
    epoch_files = glob(os.path.join(run_path, 'epoch_*.csv'))
    epoch_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    generations = []
    performance = []
    for file_path in epoch_files:
        df = pd.read_csv(file_path)
        average_fitness = df['Fitness Score'].mean()
        generation = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        generations.append(generation)
        performance.append(average_fitness)
    plt.figure(figsize=(10, 5))
    plt.plot(generations, performance, marker='o', linestyle='-', color='b')
    plt.title(f'Performance Across Generations for Run ID: {run_id}')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness Score')
    plt.grid(True)
    plt.savefig(os.path.join(run_path, 'performance_plot.png'))
    plt.show()

def monitor_runs():
    while True:
        run_ids = next(os.walk('runs'))[1]
        print("Available run IDs:", run_ids)
        selected_id = input("Enter run ID to visualize (or 'exit' to quit): ")
        if selected_id.lower() == 'exit':
            break
        elif selected_id in run_ids:
            visualize_performance(selected_id)
        else:
            print("Run ID not found. Please try again.")

if __name__ == '__main__':
    monitor_runs()
