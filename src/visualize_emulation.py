import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read emulation results
nvme_df = pd.read_csv('/root/halo/results/stats_emulation_Xeon_NVMe.csv')
sata_df = pd.read_csv('/root/halo/results/stats_emulation_Xeon_SATA.csv')

# Configure Plot Style
sns.set_theme(style="whitegrid", rc={"axes.titlesize": 14, "axes.labelsize": 12})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Xeon NVMe Distribution
sns.histplot(data=nvme_df, x='agg_speedup', kde=True, bins=25, color='#3498db', ax=ax1, label='SOTA Expected Gain')
ax1.axvline(1.0, color='#e74c3c', linestyle='-', linewidth=2, label='NATIVE Baseline (1.0x)')
ax1.axvline(1.05, color='#f1c40f', linestyle='--', linewidth=2, label='HALO Safety Threshold (1.05x)')
ax1.set_title('Target: Xeon NVMe Performance Distribution')
ax1.set_xlabel('Emulated Speedup (Target vs Baseline)')
ax1.set_ylabel('Number of Queries')
ax1.set_xlim(0.95, 1.15)
ax1.legend()

# Plot 2: Xeon SATA Distribution
sns.histplot(data=sata_df, x='agg_speedup', kde=True, bins=25, color='#2ecc71', ax=ax2, label='SOTA Expected Gain')
ax2.axvline(1.0, color='#e74c3c', linestyle='-', linewidth=2, label='NATIVE Baseline (1.0x)')
ax2.axvline(1.05, color='#f1c40f', linestyle='--', linewidth=2, label='HALO Safety Threshold (1.05x)')
ax2.set_title('Target: Xeon SATA Performance Distribution')
ax2.set_xlabel('Emulated Speedup (Target vs Baseline)')
ax2.set_ylabel('Number of Queries')
ax2.set_xlim(0.95, 1.15)
ax2.legend()

plt.tight_layout()

# Save to artifact directory
out_path = '/root/.gemini/antigravity/brain/8afbd709-f241-48e5-a53e-070dc763fe7e/emulation_visualization.png'
plt.savefig(out_path, dpi=300)
print(f"Visualization saved to {out_path}")
