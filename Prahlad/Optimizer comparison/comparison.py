import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Parse the log file and group by optimizer
sections = {}
current_optimizer = None

with open('log.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

i = 0
while i < len(lines):
    # Identify headers structured like: ############## \n OPTIMIZER_NAME \n ##############
    if lines[i].startswith('####') and i + 1 < len(lines) and not lines[i+1].startswith('####'):
        current_optimizer = lines[i+1]
        sections[current_optimizer] = []
        i += 3  # Skip past the header block
        continue
    
    # Collect numerical execution time values
    if current_optimizer and lines[i]:
        try:
            val = float(lines[i])
            sections[current_optimizer].append(val)
        except ValueError:
            pass
    i += 1

# Convert dictionary data into a pandas DataFrame
df = pd.DataFrame(sections)

# 2. Setup the visualization canvas
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Define distinct consistent colors for each optimizer
colors = {'L-BFGS-B': '#1f77b4', 'SLSQP': '#ff770f', 'BFGS': '#2ca02c', 'IPOPT': "#d12626"}

# --- Plot 1: Iteration vs Time (Raw + Rolling Mean) ---
window_size = 50
for col in df.columns:
    # Light background lines for raw iteration values
    ax1.plot(df.index, df[col], color=colors[col], alpha=0.15, linewidth=0.5)
    # Highlighted smooth line for simple moving average trends
    rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
    ax1.plot(df.index, rolling_mean, color=colors[col], label=f'{col} (50-step SMA)', linewidth=2)

ax1.set_title('MPC Execution Time over Iterations', fontsize=14, fontweight='bold')
ax1.set_xlabel('Iteration / Time Step', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlim(0, len(df))

# --- Plot 2: Boxplot Comparison of Distributions ---
sns.boxplot(data=df, ax=ax2, palette=colors, width=0.5, 
            flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
ax2.set_title('Comparison of Execution Time Distributions', fontsize=14, fontweight='bold')
ax2.set_xlabel('Optimizer', fontsize=12)
ax2.set_ylabel('Execution Time (seconds)', fontsize=12)

# Overlay Mean and Median metric annotations directly onto the boxplot
for i, col in enumerate(df.columns):
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax2.text(i, mean_val, f'Mean: {mean_val:.3f}s\nMed: {median_val:.3f}s', 
             ha='center', va='bottom', color='black', fontweight='semibold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

# Tight layout tuning and exporting the image
plt.tight_layout()
plt.savefig('mpc_optimizer_performance.png', dpi=300)
plt.close()

print("Visualization created and saved as 'mpc_optimizer_performance.png'")