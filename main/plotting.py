import pandas as pd
import matplotlib.pyplot as plt
import os

dynamic_name = "rtdetr-l_1"  # Replace with your dynamic name
csv_file = f'runs/detect/{dynamic_name}/results.csv'
save_dir = 'loss_plots'

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

plt.style.use('seaborn-darkgrid')

for column in df.columns:
    if column != 'epoch':
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df[column], label=column, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel(column)
        plt.title(f'Epoch vs {column}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{column.replace("/", "_")}.png'))

print("Plots saved successfully.")