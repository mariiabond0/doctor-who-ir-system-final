"""Generate comparison visualizations of all IR methods."""

import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load results
results_df = pd.read_csv(config.RESULTS_CSV_PATH)

# Path to save comparison chart
dw_data = config.DATA_DIR

# Extract method name (without query number)
results_df['method_clean'] = results_df['method'].str.rsplit(pat=' ', n=1).str[0]

# Group by method to get aggregate metrics
method_stats = results_df.groupby('method_clean')[['P@5', 'R@5', 'AP', 'MRR', 'nDCG']].mean()

print("\n" + "="*80)
print("COMPARISON OF ALL SEARCH METHODS")
print("="*80)
print(method_stats.round(3))
print("="*80 + "\n")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Comparison of IR Search Methods (Optimized BM25: k1=0.4, b=1.0)', fontsize=16, fontweight='bold')

metrics = ['P@5', 'R@5', 'AP', 'MRR', 'nDCG']
colors = sns.color_palette("husl", len(method_stats))

# Individual metric bar plots
for idx, metric in enumerate(metrics):
    ax = axes.flat[idx]
    method_stats[metric].sort_values(ascending=False).plot(
        kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.2
    )
    ax.set_title(f'Mean {metric}', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(method_stats[metric].sort_values(ascending=False)):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Overall score (radar-style normalized comparison on 6th plot)
ax = axes.flat[5]
# Normalize metrics to 0-1 scale for overall score
normalized = method_stats.copy()
for col in normalized.columns:
    normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

# Calculate overall score as mean of normalized metrics
overall_score = normalized.mean(axis=1).sort_values(ascending=False)
overall_score.plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='black', linewidth=1.2)
ax.set_title('Overall Normalized Score', fontweight='bold', fontsize=12)
ax.set_xlabel('Score', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(overall_score):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(dw_data/'method_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Comparison chart saved to: dw_data/method_comparison.png\n")

# Create detailed ranking table
print("DETAILED RANKING (by nDCG):")
print("-" * 80)
ranking = method_stats.sort_values('nDCG', ascending=False).reset_index()
ranking['Rank'] = range(1, len(ranking) + 1)
ranking = ranking[['Rank', 'method_clean', 'P@5', 'R@5', 'AP', 'MRR', 'nDCG']]
ranking.columns = ['Rank', 'Method', 'P@5', 'R@5', 'AP', 'MRR', 'nDCG']

for idx, row in ranking.iterrows():
    print(f"{int(row['Rank']):2d}. {row['Method']:30s} | "
          f"P@5: {row['P@5']:.3f} | R@5: {row['R@5']:.3f} | AP: {row['AP']:.3f} | "
          f"MRR: {row['MRR']:.3f} | nDCG: {row['nDCG']:.3f}")
print("-" * 80)

# Save ranking to CSV
ranking.to_csv(dw_data / 'method_ranking.csv', index=False)
print(f"✓ Ranking saved to: dw_data/method_ranking.csv\n")

# Per-method statistics
print("\nPER-QUERY STATISTICS BY METHOD:")
print("-" * 80)
for method in method_stats.index:
    method_data = results_df[results_df['method_clean'] == method]
    print(f"\n{method}:")
    print(f"  Queries tested: {len(method_data)}")
    print(f"  Mean overlap: {method_data['overlap'].mean():.2f}")
    print(f"  Min/Max nDCG: {method_data['nDCG'].min():.3f} / {method_data['nDCG'].max():.3f}")
    print(f"  nDCG Std Dev: {method_data['nDCG'].std():.3f}")

print("\n" + "="*80)
