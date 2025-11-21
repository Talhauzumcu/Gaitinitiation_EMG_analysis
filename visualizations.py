#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from utils import *
# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

#%%
SAVE_DIR = f'./plots'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
df = pd.read_csv('emg_analysis_results.csv')
df = df[(df['reaction_time(ms)'] > 0) & (df['reaction_time(ms)'] < 1000)]

cols = df.select_dtypes('number').columns.drop(['subject_id','reaction_time(ms)'])  
df_sub = df.loc[:, cols]
lim = np.abs((df_sub - df_sub.mean()) / df_sub.std(ddof=0)) < 3
df.loc[:, cols] = df_sub.where(lim, np.nan)

df_oa = df[df['category'] == 'OA']
df_ya = df[df['category'] == 'YA']
#%% 1. EMG AMPLITUDE COMPARISON ACROSS TEMPORAL WINDOWS YA (with jittered data points)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EMG Amplitude Comparison: Young Adults success vs non success', fontsize=16, fontweight='bold')

amplitude_columns = [
    'amplitude_mean(preGreen_copOnset)',
    'amplitude_mean(copOnset_stopPerceived)',
    'amplitude_mean(stopPerceived_postPeak)',
    'amplitude_mean(copOnset_postPeak)',
    'amplitude_mean(postPeak_frontPeak)'
]

phase_labels = [
    'Pre-Green to COP Onset',
    'COP Onset to Stop Perceived',
    'Stop Perceived to Post Peak',
    'COP Onset to Post Peak',
    'Post Peak to Front Peak'
]

for idx, (col, label) in enumerate(zip(amplitude_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers (so overlaid points are clear)
    sns.boxplot(data=df_ya, x='emg_channel', y=col, hue='success',
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=df_ya, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.5, size=3,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries (created by both boxplot and stripplot)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Mean Amplitude')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(amplitude_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_amplitude_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 2. EMG AMPLITUDE COMPARISON ACROSS TEMPORAL WINDOWS OA (with jittered data points)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EMG Amplitude Comparison: Young Adults success vs non success', fontsize=16, fontweight='bold')

amplitude_columns = [
    'amplitude_mean(preGreen_copOnset)',
    'amplitude_mean(copOnset_stopPerceived)',
    'amplitude_mean(stopPerceived_postPeak)',
    'amplitude_mean(copOnset_postPeak)',
    'amplitude_mean(postPeak_frontPeak)'
]

phase_labels = [
    'Pre-Green to COP Onset',
    'COP Onset to Stop Perceived',
    'Stop Perceived to Post Peak',
    'COP Onset to Post Peak',
    'Post Peak to Front Peak'
]

for idx, (col, label) in enumerate(zip(amplitude_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers (so overlaid points are clear)
    sns.boxplot(data=df_oa, x='emg_channel', y=col, hue='success',
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=df_oa, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.5, size=3,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries (created by both boxplot and stripplot)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Mean Amplitude')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(amplitude_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_amplitude_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 3. EMG AMPLITUDE COMPARISON ACROSS TEMPORAL WINDOWS OA vs YA(with jittered data points)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EMG Amplitude Comparison: Young Adults success vs non success', fontsize=16, fontweight='bold')

amplitude_columns = [
    'amplitude_mean(preGreen_copOnset)',
    'amplitude_mean(copOnset_stopPerceived)',
    'amplitude_mean(stopPerceived_postPeak)',
    'amplitude_mean(copOnset_postPeak)',
    'amplitude_mean(postPeak_frontPeak)'
]

phase_labels = [
    'Pre-Green to COP Onset',
    'COP Onset to Stop Perceived',
    'Stop Perceived to Post Peak',
    'COP Onset to Post Peak',
    'Post Peak to Front Peak'
]

for idx, (col, label) in enumerate(zip(amplitude_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers (so overlaid points are clear)
    sns.boxplot(data=df, x='emg_channel', y=col, hue='category',
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=df, x='emg_channel', y=col, hue='category',
                  dodge=True, jitter=0.25, alpha=0.5, size=3,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries (created by both boxplot and stripplot)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Category')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Mean Amplitude')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(amplitude_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_amplitude_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 4. iEMG COMPARISON - YA success vs non-success (Violin + Box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Integrated EMG (iEMG) Comparison: Young Adults success vs non-success', fontsize=16, fontweight='bold')

iemg_columns = [
    'iEMG(preGreen_copOnset)',
    'iEMG(copOnset_stopPerceived)',
    'iEMG(stopPerceived_postPeak)',
    'iEMG(copOnset_postPeak)',
    'iEMG(postPeak_frontPeak)'
]

phase_labels = [
    'Pre-Green to COP Onset',
    'COP Onset to Stop Perceived',
    'Stop Perceived to Post Peak',
    'COP Onset to Post Peak',
    'Post Peak to Front Peak'
]

for idx, (col, label) in enumerate(zip(iemg_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=df_ya, x='emg_channel', y=col, hue='success',
                   split=True, ax=ax, inner=None, palette=['coral', 'skyblue'], alpha=.8)
    
    sns.stripplot(data=df_ya, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.4, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    # Keep only first 2 (from violin plot)
    ax.legend(handles[:2], labels[:2], title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('iEMG (µV·s)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(iemg_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/iemg_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 5. iEMG COMPARISON - OA success vs non-success 
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Integrated EMG (iEMG) Comparison: Older Adults success vs non-success', fontsize=16, fontweight='bold')

iemg_columns = [
    'iEMG(preGreen_copOnset)',
    'iEMG(copOnset_stopPerceived)',
    'iEMG(stopPerceived_postPeak)',
    'iEMG(copOnset_postPeak)',
    'iEMG(postPeak_frontPeak)'
]

for idx, (col, label) in enumerate(zip(iemg_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=df_oa, x='emg_channel', y=col, hue='success',
                   split=True, ax=ax, inner=None, palette=['coral', 'skyblue'], alpha=.8)
    
    sns.stripplot(data=df_oa, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.4, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('iEMG (µV·s)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(iemg_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/iemg_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 6. iEMG COMPARISON - OA vs YA (Violin + Box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Integrated EMG (iEMG) Comparison: Older Adults vs Young Adults', fontsize=16, fontweight='bold')

iemg_columns = [
    'iEMG(preGreen_copOnset)',
    'iEMG(copOnset_stopPerceived)',
    'iEMG(stopPerceived_postPeak)',
    'iEMG(copOnset_postPeak)',
    'iEMG(postPeak_frontPeak)'
]

for idx, (col, label) in enumerate(zip(iemg_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=df, x='emg_channel', y=col, hue='category',
                   split=True, ax=ax, inner=None, palette=['coral', 'skyblue'], alpha=.8)
    
    sns.stripplot(data=df, x='emg_channel', y=col, hue='category',
                  dodge=True, jitter=0.25, alpha=0.4, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Category')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('iEMG (µV·s)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(iemg_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/iemg_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 7. AVERAGE PEAKS COMPARISON - YA success vs non-success (Violin + Box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Average Peak EMG Comparison: Young Adults success vs non-success', fontsize=16, fontweight='bold')

peak_columns = [
    'average_peaks(preGreen_copOnset)',
    'average_peaks(copOnset_stopPerceived)',
    'average_peaks(stopPerceived_postPeak)',
    'average_peaks(copOnset_postPeak)',
    'average_peaks(postPeak_frontPeak)'
]

for idx, (col, label) in enumerate(zip(peak_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.barplot(data=df_ya, x='emg_channel', y=col, hue='success',
                   ax=ax, palette=['coral', 'skyblue'], alpha=1)
    
    sns.stripplot(data=df_ya, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.7, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Average Peak Amplitude (µV)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(peak_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/average_peaks_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 8. AVERAGE PEAKS COMPARISON - OA success vs non-success (Violin + Box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Average Peak EMG Comparison: Older Adults success vs non-success', fontsize=16, fontweight='bold')

peak_columns = [
    'average_peaks(preGreen_copOnset)',
    'average_peaks(copOnset_stopPerceived)',
    'average_peaks(stopPerceived_postPeak)',
    'average_peaks(copOnset_postPeak)',
    'average_peaks(postPeak_frontPeak)'
]

for idx, (col, label) in enumerate(zip(peak_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.barplot(data=df_oa, x='emg_channel', y=col, hue='success',
                   ax=ax, palette=['coral', 'skyblue'], alpha=1)
    
    sns.stripplot(data=df_oa, x='emg_channel', y=col, hue='success',
                  dodge=True, jitter=0.25, alpha=0.7, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Success')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Average Peak Amplitude (µV)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(peak_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/average_peaks_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 9. AVERAGE PEAKS COMPARISON - OA vs YA (Violin + Box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Average Peak EMG Comparison: Older Adults vs Young Adults', fontsize=16, fontweight='bold')

peak_columns = [
    'average_peaks(preGreen_copOnset)',
    'average_peaks(copOnset_stopPerceived)',
    'average_peaks(stopPerceived_postPeak)',
    'average_peaks(copOnset_postPeak)',
    'average_peaks(postPeak_frontPeak)'
]

for idx, (col, label) in enumerate(zip(peak_columns, phase_labels)):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.barplot(data=df, x='emg_channel', y=col, hue='category',
                   ax=ax, palette=['coral', 'skyblue'], alpha=1)
    
    sns.stripplot(data=df, x='emg_channel', y=col, hue='category',
                  dodge=True, jitter=0.25, alpha=0.7, size=2,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Category')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Average Peak Amplitude (µV)')
    ax.tick_params(axis='x', rotation=45)

# Remove extra subplot
if len(peak_columns) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/average_peaks_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()


# %%
cocontraction_df = pd.read_csv('emg_cocontraction_results.csv')
cocontraction_df = cocontraction_df[(cocontraction_df['reaction_time(ms)'] > 0) & (cocontraction_df['reaction_time(ms)'] < 1000)]

#Remove outliers based on z-score
# cols = cocontraction_df.select_dtypes('number').columns.drop(['Subject_ID','reaction_time(ms)'])  
# cocontraction_df_sub = cocontraction_df.loc[:, cols]
# lim = np.abs((cocontraction_df_sub - cocontraction_df_sub.mean()) / cocontraction_df_sub.std(ddof=0)) < 3
# cocontraction_df.loc[:, cols] = cocontraction_df_sub.where(lim, np.nan)
# Separate by category
cocontraction_df_oa = cocontraction_df[cocontraction_df['category'] == 'OA']
cocontraction_df_ya = cocontraction_df[cocontraction_df['category'] == 'YA']

#%% 10. COCONTRACTION COMPARISON - YA success vs non-success
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Young Adults success vs non-success', fontsize=16, fontweight='bold')

cocontraction_columns = cocontraction_df.columns[5:]

cocontraction_labels = [
    'Right Tib Ant - Right Soleus | Green to frontal Peak',
    'Right Tib Ant - Right Soleus | CoP onset to Stop signal',
    'Right Tib Ant - Right Soleus | Stop signal to Post Peak',

    'Left Tib Ant - Left Soleus | Green to frontal Peak',
    'Left Tib Ant - Left Soleus | CoP onset to Stop signal',
    'Left Tib Ant - Left Soleus | Stop signal to Post Peak',

    'Right Tib Ant - Right Gastroc Med | Green to frontal Peak',
    'Right Tib Ant - Right Gastroc Med | CoP onset to Stop signal',
    'Right Tib Ant - Right Gastroc Med | Stop signal to Post Peak',
    
    'Left Tib Ant - Left Gastroc Med | Green to frontal Peak',
    'Left Tib Ant - Left Gastroc Med | CoP onset to Stop signal',
    'Left Tib Ant - Left Gastroc Med | Stop signal to Post Peak'
]

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers
    sns.boxplot(data=cocontraction_df_ya, x='success', y=col,
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=cocontraction_df_ya, x='success', y=col,
                  jitter=0.25, alpha=0.5, size=4,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Success')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 11. COCONTRACTION COMPARISON - OA success vs non-success
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Older Adults success vs non-success', fontsize=16, fontweight='bold')

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers
    sns.boxplot(data=cocontraction_df_oa, x='success', y=col,
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=cocontraction_df_oa, x='success', y=col,
                  jitter=0.25, alpha=0.5, size=4,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Success')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 12. COCONTRACTION COMPARISON - OA vs YA
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Older Adults vs Young Adults', fontsize=16, fontweight='bold')

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Boxplot without showing fliers
    sns.boxplot(data=cocontraction_df, x='category', y=col,
                ax=ax, showfliers=False, palette=['coral', 'skyblue'])
    
    # Jittered individual data points on top of the boxplot
    sns.stripplot(data=cocontraction_df, x='category', y=col,
                  jitter=0.25, alpha=0.5, size=4,
                  palette=['coral', 'skyblue'], ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Category')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 13. COCONTRACTION COMPARISON - YA success vs non-success (Violin plots)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Young Adults success vs non-success', fontsize=16, fontweight='bold')

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=cocontraction_df_ya, x='success', y=col,
                   split=True, gap=-0.2, ax=ax, inner='quartile', palette=['coral', 'skyblue'])
    
    sns.stripplot(data=cocontraction_df_ya, x='success', y=col,
                  jitter=0.25, alpha=0.4, size=3,
                  color='black', ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Success')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_violin_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 14. COCONTRACTION COMPARISON - OA success vs non-success (Violin plots)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Older Adults success vs non-success', fontsize=16, fontweight='bold')

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=cocontraction_df_oa, x='success', y=col,
                   split=True, gap=-0.2, ax=ax, inner='quartile', palette=['coral', 'skyblue'])
    
    sns.stripplot(data=cocontraction_df_oa, x='success', y=col,
                  jitter=0.25, alpha=0.4, size=3,
                  color='black', ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Success')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_violin_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% 15. COCONTRACTION COMPARISON - OA vs YA (Violin plots)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Muscle Cocontraction Comparison: Older Adults vs Young Adults', fontsize=16, fontweight='bold')

for idx, (col, label) in enumerate(zip(cocontraction_columns, cocontraction_labels)):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Violin plot for distribution
    sns.violinplot(data=cocontraction_df, x='category', y=col,
                   split=True, gap=-0.2, ax=ax, inner='quartile', palette=['coral', 'skyblue'])
    
    sns.stripplot(data=cocontraction_df, x='category', y=col,
                  jitter=0.25, alpha=0.4, size=3,
                  color='black', ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('Category')
    ax.set_ylabel('Cocontraction Percentage')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/cocontraction_violin_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% Time series plots of EMG signals between OA and YA groups
subjects = load_subjects_pickle("subjects_cache.pkl")
# %%
data_list = []
for subject in subjects:
    for trial in subject.trials.values():
        for EMG in trial.emgs.values():
            try:
                emg_data = EMG.processed_data
                start_idx = trial.events['CoP_onset']
                end_idx = trial.events['frontal_peak']
                emg_data = emg_data[start_idx:end_idx]
                # time normalize the emg data to 0-100%
                original_length = len(emg_data)
                original_time = np.linspace(0, 100, original_length)
                normalized_time = np.linspace(0, 100, 200)
                emg_data_normalized = np.interp(normalized_time, original_time, emg_data)
                category = 'YA' if subject.is_young else 'OA'
                data_list.append({
                    'category': category,
                    'emg_channel': EMG.name,
                    'success': trial.success,
                    'data': emg_data_normalized})
            except Exception as e:
                print(f"Missing data for Subject {subject.subject_id},"
                      f"Trial {trial.trial_name}, EMG {EMG.name} e: {e}")
                continue
# %%
time_series_df = pd.DataFrame(data_list)

#%% Plot mean time series with 95% CI for each EMG channel (YA vs OA)
# Get unique EMG channels
emg_channels = time_series_df['emg_channel'].unique()

# Create time vector (0-100%)
time_normalized = np.linspace(0, 100, 200)

# Calculate number of subplots needed
n_channels = len(emg_channels)
n_cols = 3
n_rows = int(np.ceil(n_channels / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_channels > 1 else [axes]

fig.suptitle('Mean EMG Time Series (CoP Onset to Frontal Peak): YA vs OA with 95% CI', 
             fontsize=16, fontweight='bold')

for idx, channel in enumerate(emg_channels):
    ax = axes[idx]
    
    channel_data = time_series_df[time_series_df['emg_channel'] == channel]
    
    ya_data = channel_data[channel_data['category'] == 'YA']['data'].values
    oa_data = channel_data[channel_data['category'] == 'OA']['data'].values
    
    ya_array = np.vstack(ya_data) if len(ya_data) > 0 else np.array([])
    oa_array = np.vstack(oa_data) if len(oa_data) > 0 else np.array([])
    
    if len(ya_array) > 0:
        ya_mean = np.mean(ya_array, axis=0)
        ya_std = np.std(ya_array, axis=0)
        ya_n = len(ya_array)
        ya_sem = ya_std / np.sqrt(ya_n)  # Standard error of mean
        ya_ci = 1.96 * ya_sem  # 95% CI
        
        # Plot YA
        ax.plot(time_normalized, ya_mean, color='skyblue', linewidth=2.5, label=f'YA (n={ya_n})')
        ax.fill_between(time_normalized, ya_mean - ya_ci, ya_mean + ya_ci, 
                        color='skyblue', alpha=0.3)
    
    if len(oa_array) > 0:
        oa_mean = np.mean(oa_array, axis=0)
        oa_std = np.std(oa_array, axis=0)
        oa_n = len(oa_array)
        oa_sem = oa_std / np.sqrt(oa_n)  # Standard error of mean
        oa_ci = 1.96 * oa_sem  # 95% CI
        
        # Plot OA
        ax.plot(time_normalized, oa_mean, color='coral', linewidth=2.5, label=f'OA (n={oa_n})')
        ax.fill_between(time_normalized, oa_mean - oa_ci, oa_mean + oa_ci, 
                        color='coral', alpha=0.3)
    
    ax.set_title(channel, fontweight='bold')
    ax.set_xlabel('Time (% of COP onset to Frontal Peak)')
    ax.set_ylabel('EMG Amplitude (normalized)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

for idx in range(n_channels, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_timeseries_mean_CI_YA_vs_OA.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% Plot mean time series with 95% CI for each EMG channel (YA: success vs non-success)
# Get unique EMG channels
emg_channels = time_series_df['emg_channel'].unique()

# Create time vector (0-100%)
time_normalized = np.linspace(0, 100, 200)

# Calculate number of subplots needed
n_channels = len(emg_channels)
n_cols = 3
n_rows = int(np.ceil(n_channels / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_channels > 1 else [axes]

fig.suptitle('Mean EMG Time Series - Young Adults: Success vs Non-Success with 95% CI', 
             fontsize=16, fontweight='bold')

for idx, channel in enumerate(emg_channels):
    ax = axes[idx]
    
    channel_data = time_series_df[(time_series_df['emg_channel'] == channel) & 
                                   (time_series_df['category'] == 'YA')]
    
    success_data = channel_data[channel_data['success'] == True]['data'].values
    non_success_data = channel_data[channel_data['success'] == False]['data'].values
    
    success_array = np.vstack(success_data) if len(success_data) > 0 else np.array([])
    non_success_array = np.vstack(non_success_data) if len(non_success_data) > 0 else np.array([])
    
    if len(success_array) > 0:
        success_mean = np.mean(success_array, axis=0)
        success_std = np.std(success_array, axis=0)
        success_n = len(success_array)
        success_sem = success_std / np.sqrt(success_n)
        success_ci = 1.96 * success_sem
        
        ax.plot(time_normalized, success_mean, color='green', linewidth=2.5, 
                label=f'Success (n={success_n})')
        ax.fill_between(time_normalized, success_mean - success_ci, success_mean + success_ci, 
                        color='green', alpha=0.3)
    
    if len(non_success_array) > 0:
        non_success_mean = np.mean(non_success_array, axis=0)
        non_success_std = np.std(non_success_array, axis=0)
        non_success_n = len(non_success_array)
        non_success_sem = non_success_std / np.sqrt(non_success_n)
        non_success_ci = 1.96 * non_success_sem
        
        ax.plot(time_normalized, non_success_mean, color='red', linewidth=2.5, 
                label=f'Non-Success (n={non_success_n})')
        ax.fill_between(time_normalized, non_success_mean - non_success_ci, 
                        non_success_mean + non_success_ci, 
                        color='red', alpha=0.3)
    
    ax.set_title(channel, fontweight='bold')
    ax.set_xlabel('Time (% of COP onset to Frontal Peak)')
    ax.set_ylabel('EMG Amplitude (normalized)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

for idx in range(n_channels, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_timeseries_mean_CI_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% Plot mean time series with 95% CI for each EMG channel (OA: success vs non-success)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_channels > 1 else [axes]

fig.suptitle('Mean EMG Time Series - Older Adults: Success vs Non-Success with 95% CI', 
             fontsize=16, fontweight='bold')

for idx, channel in enumerate(emg_channels):
    ax = axes[idx]
    
    channel_data = time_series_df[(time_series_df['emg_channel'] == channel) & 
                                   (time_series_df['category'] == 'OA')]
    
    success_data = channel_data[channel_data['success'] == True]['data'].values
    non_success_data = channel_data[channel_data['success'] == False]['data'].values
    
    success_array = np.vstack(success_data) if len(success_data) > 0 else np.array([])
    non_success_array = np.vstack(non_success_data) if len(non_success_data) > 0 else np.array([])

    if len(success_array) > 0:
        success_mean = np.mean(success_array, axis=0)
        success_std = np.std(success_array, axis=0)
        success_n = len(success_array)
        success_sem = success_std / np.sqrt(success_n)
        success_ci = 1.96 * success_sem
        
        ax.plot(time_normalized, success_mean, color='green', linewidth=2.5, 
                label=f'Success (n={success_n})')
        ax.fill_between(time_normalized, success_mean - success_ci, success_mean + success_ci, 
                        color='green', alpha=0.3)
    
    if len(non_success_array) > 0:
        non_success_mean = np.mean(non_success_array, axis=0)
        non_success_std = np.std(non_success_array, axis=0)
        non_success_n = len(non_success_array)
        non_success_sem = non_success_std / np.sqrt(non_success_n)
        non_success_ci = 1.96 * non_success_sem
        
        ax.plot(time_normalized, non_success_mean, color='red', linewidth=2.5, 
                label=f'Non-Success (n={non_success_n})')
        ax.fill_between(time_normalized, non_success_mean - non_success_ci, 
                        non_success_mean + non_success_ci, 
                        color='red', alpha=0.3)
    
    ax.set_title(channel, fontweight='bold')
    ax.set_xlabel('Time (% of COP onset to Frontal Peak)')
    ax.set_ylabel('EMG Amplitude (normalized)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

for idx in range(n_channels, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_timeseries_mean_CI_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
# plt.show()

# %%
