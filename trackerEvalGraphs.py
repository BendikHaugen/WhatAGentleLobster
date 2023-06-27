import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/tracker_results/LOBSTER-test/model_summary.csv')

metrics = ['HOTA', 'MOTA', 'MOTP', 'Recall', 'Precision', 'IDsw']
metrics2 = ['HOTA', 'MOTA', 'IDF1']

for metric in metrics2:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=df)
    plt.title(f'')
    plt.xticks(rotation=90) 
    min_val, max_val = df[metric].min(), df[metric].max()
    plt.ylim(min_val*0.95, min(max_val*1.05, 1))
    plt.savefig(f"{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()  


corr_matrix = df[metrics].corr()

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1)

# Save the heatmap
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.close()