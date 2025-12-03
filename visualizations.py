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

SAVE_DIR = f'./plots'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
#%%

    
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

# %% AVERAGE ON OFF SIGNAL PLOTS FOR YA AND OA
on_off_df = pd.read_csv('emg_on_off_signals_START_pre_green_END_frontalpeak.csv')
on_off_df = on_off_df[(on_off_df['reaction_time'] > 0) & (on_off_df['reaction_time'] < 1000)]
time_cols = [col for col in on_off_df.columns if col.startswith('Time_')]
ya_means = on_off_df[on_off_df['category'] == 'YA'].groupby('emg_channel')[time_cols].mean()
oa_means = on_off_df[on_off_df['category'] == 'OA'].groupby('emg_channel')[time_cols].mean()
ya_early_means = on_off_df[(on_off_df['category'] == 'YA') & (on_off_df['latency'] == 'early')].groupby('emg_channel')[time_cols].mean()
oa_early_means = on_off_df[(on_off_df['category'] == 'OA') & (on_off_df['latency'] == 'early')].groupby('emg_channel')[time_cols].mean()
ya_late_means = on_off_df[(on_off_df['category'] == 'YA') & (on_off_df['latency'] == 'late')].groupby('emg_channel')[time_cols].mean()
oa_late_means = on_off_df[(on_off_df['category'] == 'OA') & (on_off_df['latency'] == 'late')].groupby('emg_channel')[time_cols].mean()

#%%
def add_spacing(data, spacing=1):
    """Insert white (0) rows between each EMG channel row"""
    n_channels, n_timepoints = data.shape
    # Create new array with spacing rows (filled with NaN to show as white)
    spaced_data = np.full((n_channels + (n_channels - 1) * spacing, n_timepoints), np.nan)
    for i in range(n_channels):
        spaced_data[i * (spacing + 1), :] = data[i, :]
    return spaced_data
# %% OVERALL MEAN ON OFF SIGNAL PLOTS FOR YA AND OA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

spacing = 1  # Number of white rows between channels
# YA group plot
ya_data = ya_means.values
ya_spaced = add_spacing(ya_data, spacing)
ax1 = axes[0]
im1 = ax1.imshow(ya_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
# Adjust yticks to account for spacing
ytick_positions = [i * (spacing + 1) for i in range(len(ya_means.index))]
ax1.set_yticks(ytick_positions)
ax1.set_yticklabels(ya_means.index)
ax1.set_xlabel('Time (%)')
ax1.set_ylabel('EMG Channel')
ax1.set_title('YA Group - Mean EMG On/Off')
ax1.set_xticks(np.linspace(0, ya_data.shape[1]-1, 6))
ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax1.grid(False)

# OA group plot
oa_data = oa_means.values
oa_spaced = add_spacing(oa_data, spacing)
ax2 = axes[1]
im2 = ax2.imshow(oa_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ytick_positions = [i * (spacing + 1) for i in range(len(oa_means.index))]
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(oa_means.index)
ax2.set_xlabel('Time (%)')
ax2.set_ylabel('EMG Channel')
ax2.set_title('OA Group - Mean EMG On/Off')
ax2.set_xticks(np.linspace(0, oa_data.shape[1]-1, 6))
ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax2.grid(False)

# Add colorbars
plt.colorbar(im1, ax=ax1, label='Mean On/off pct')
plt.colorbar(im2, ax=ax2, label='Mean On/off pct')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_on_off_signals_YA_vs_OA.png', dpi=300, bbox_inches='tight')
plt.show()
# %%Mean of on off signals for early and late latency trials
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

spacing = 1  # Number of white rows between channels
# YA early group plot
ya_early_data = ya_early_means.values
ya_spaced = add_spacing(ya_early_data, spacing)
ax1 = axes[0, 0]
im1 = ax1.imshow(ya_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
# Adjust yticks to account for spacing
ytick_positions = [i * (spacing + 1) for i in range(len(ya_early_means.index))]
ax1.set_yticks(ytick_positions)
ax1.set_yticklabels(ya_early_means.index)
ax1.set_xlabel('Time (%)')
ax1.set_ylabel('EMG Channel')
ax1.set_title('YA Group - Mean EMG On/Off Early')
ax1.set_xticks(np.linspace(0, ya_early_data.shape[1]-1, 6))
ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax1.grid(False)

# OA early group plot
oa_early_data = oa_early_means.values
oa_spaced = add_spacing(oa_early_data, spacing)
ax2 = axes[0, 1]
im2 = ax2.imshow(oa_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ytick_positions = [i * (spacing + 1) for i in range(len(oa_early_means.index))]
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(oa_early_means.index)
ax2.set_xlabel('Time (%)')
ax2.set_ylabel('EMG Channel')
ax2.set_title('OA Group - Mean EMG On/Off early')
ax2.set_xticks(np.linspace(0, oa_early_data.shape[1]-1, 6))
ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax2.grid(False)

# YA late group plot
ya_late_data = ya_late_means.values
ya_spaced = add_spacing(ya_late_data, spacing)
ax1 = axes[1, 0]
im1 = ax1.imshow(ya_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
# Adjust yticks to account for spacing
ytick_positions = [i * (spacing + 1) for i in range(len(ya_late_means.index))]
ax1.set_yticks(ytick_positions)
ax1.set_yticklabels(ya_late_means.index)
ax1.set_xlabel('Time (%)')
ax1.set_ylabel('EMG Channel')
ax1.set_title('YA Group - Mean EMG On/Off Late')
ax1.set_xticks(np.linspace(0, ya_late_data.shape[1]-1, 6))
ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax1.grid(False)

# OA late group plot
oa_late_data = oa_late_means.values
oa_spaced = add_spacing(oa_late_data, spacing)
ax2 = axes[1, 1]
im2 = ax2.imshow(oa_spaced, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
ytick_positions = [i * (spacing + 1) for i in range(len(oa_late_means.index))]
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(oa_late_means.index)
ax2.set_xlabel('Time (%)')
ax2.set_ylabel('EMG Channel')
ax2.set_title('OA Group - Mean EMG On/Off Late')
ax2.set_xticks(np.linspace(0, oa_late_data.shape[1]-1, 6))
ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax2.grid(False)


plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_on_off_signals_YA_vs_OA_wlatency.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
def add_spacing_interleaved(ya_data, oa_data, spacing=1):
    """Interleave YA and OA rows for each EMG channel with spacing between channel groups"""
    n_channels, n_timepoints = ya_data.shape
    # Each channel group has: YA row, OA row, then spacing rows
    rows_per_channel = 2 + spacing
    total_rows = n_channels * rows_per_channel - spacing  # No spacing after last channel
    
    spaced_data = np.full((total_rows, n_timepoints), np.nan)
    
    for i in range(n_channels):
        base_row = i * rows_per_channel
        spaced_data[base_row, :] = ya_data[i, :]      # YA row
        spaced_data[base_row + 1, :] = oa_data[i, :]  # OA row
        # spacing rows remain NaN (white)
    
    return spaced_data

#%% interlaved with latency
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
spacing = 1

# Early: Interleaved YA vs OA
ya_early_data = ya_early_means.values
oa_early_data = oa_early_means.values
interleaved_early = add_spacing_interleaved(ya_early_data, oa_early_data, spacing)

ax1 = axes[0]
im1 = ax1.imshow(interleaved_early, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

# Set yticks at channel groups, label with channel names
rows_per_channel = 2 + spacing
ytick_positions = [i * rows_per_channel + 0.5 for i in range(len(ya_early_means.index))]  # Center between YA/OA
ax1.set_yticks(ytick_positions)
ax1.set_yticklabels(ya_early_means.index)
ax1.set_xlabel('Time (%)')
ax1.set_ylabel('EMG Channel')
ax1.set_title('Early - YA (top) vs OA (bottom) per channel')
ax1.set_xticks(np.linspace(0, ya_early_data.shape[1]-1, 6))
ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax1.grid(False)

# Late: Interleaved YA vs OA
ya_late_data = ya_late_means.values
oa_late_data = oa_late_means.values
interleaved_late = add_spacing_interleaved(ya_late_data, oa_late_data, spacing)

ax2 = axes[1]
im2 = ax2.imshow(interleaved_late, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

ytick_positions = [i * rows_per_channel + 0.5 for i in range(len(ya_late_means.index))]
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(ya_late_means.index)
ax2.set_xlabel('Time (%)')
ax2.set_ylabel('EMG Channel')
ax2.set_title('Late - YA (top) vs OA (bottom) per channel')
ax2.set_xticks(np.linspace(0, ya_late_data.shape[1]-1, 6))
ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax2.grid(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_on_off_signals_YA_vs_OA_wlatency_interleaved.png', dpi=300, bbox_inches='tight')
plt.show()

#%% interlaved withiout latency
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
spacing = 1

# Early: Interleaved YA vs OA
ya_data = ya_means.values
oa_data = oa_means.values
interleaved = add_spacing_interleaved(ya_data, oa_data, spacing)
im = ax.imshow(interleaved, cmap='gray_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

# Set yticks at channel groups, label with channel names
rows_per_channel = 2 + spacing
ytick_positions = [i * rows_per_channel + 0.5 for i in range(len(ya_means.index))]  # Center between YA/OA
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ya_means.index)
ax.set_xlabel('Time (%)')
ax.set_ylabel('EMG Channel')
ax.set_title('Early - YA (top) vs OA (bottom) per channel')
ax.set_xticks(np.linspace(0, ya_data.shape[1]-1, 6))
ax.set_xticklabels(['0', '20', '40', '60', '80', '100'])
ax.grid(False)

plt.colorbar(im, ax=ax, label='Mean On/off pct')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emg_on_off_signals_YA_vs_OA_interleaved.png', dpi=300, bbox_inches='tight')
plt.show()

#%% EMG On/Off Horizontal Boxplots - YA vs OA comparison (Green signal as reference)
csv_path='emg_on_off_green_absolute.csv'

# Load data
df = pd.read_csv(csv_path)
df = df[(df['reaction_time'] > 0) & (df['reaction_time'] < 1000)]
# Calculate times relative to green signal
df['on_relative'] = df['on'] - df['green']
df['off_relative'] = df['off'] - df['green']
df['cop_onset_relative'] = df['cop_onset'] - df['green']
df['frontal_peak_relative'] = df['frontal_peak'] - df['green']

# Get unique EMG channels and sort them
emg_channels = df['emg_channel'].unique()
emg_channels = sorted(emg_channels)

# Create short names for EMG channels
channel_short_names = {
    '01_ri_soleus': 'R Sol',
    '02_ri_gastroc_med': 'R GMed',
    '03_ri_tib_ant': 'R TA',
    '06_le_tib_ant': 'L TA',
    '07_le_soleus': 'L Sol',
    '08_le_gastroc_med': 'L GMed'
}

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 10))

# Colors for YA and OA
colors = {'YA': '#3498db', 'OA': '#e74c3c'}  # Blue for YA, Red for OA

# Position parameters
n_channels = len(emg_channels)
channel_height = 1.0  # Height allocated per channel
box_height = 0.35     # Height of each box
gap_between_categories = 0.05  # Gap between YA and OA boxes

y_positions = []
y_labels = []

# Stats for annotations
stats_data = []

for i, channel in enumerate(emg_channels):
    channel_data = df[df['emg_channel'] == channel]
    base_y = (n_channels - 1 - i) * channel_height  # Reverse order so first channel is at top
    
    for cat_idx, category in enumerate(['YA', 'OA']):
        cat_data = channel_data[channel_data['category'] == category]
        
        # Calculate y position for this category
        if cat_idx == 0:  # YA - upper position
            y_pos = base_y + gap_between_categories + box_height/2
        else:  # OA - lower position
            y_pos = base_y - gap_between_categories - box_height/2
        
        # Get on and off data
        on_data = cat_data['on_relative'].dropna()
        off_data = cat_data['off_relative'].dropna()
        
        # Plot ON boxplot (filled box)
        if len(on_data) > 0:
            bp_on = ax.boxplot([on_data], positions=[y_pos + 0.08], vert=False, 
                                widths=box_height * 0.4, patch_artist=True,
                                boxprops=dict(facecolor=colors[category], alpha=0.7, linewidth=1.5),
                                medianprops=dict(color='black', linewidth=2),
                                whiskerprops=dict(color=colors[category], linewidth=1.5),
                                capprops=dict(color=colors[category], linewidth=1.5),
                                flierprops=dict(marker='o', markerfacecolor=colors[category], 
                                                markersize=3, alpha=0.5))
            
            # Add median annotation for ON
            # median_on = np.median(on_data) * 1000  # Convert to ms
            # ax.text(np.median(on_data), y_pos + 0.08 + box_height * 0.25, 
            #         f'{int(median_on)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot OFF boxplot (unfilled box with dashed lines)
        if len(off_data) > 0:
            bp_off = ax.boxplot([off_data], positions=[y_pos - 0.08], vert=False,
                                widths=box_height * 0.4, patch_artist=True,
                                boxprops=dict(facecolor='white', edgecolor=colors[category], 
                                                linewidth=1.5, linestyle='--'),
                                medianprops=dict(color='black', linewidth=2),
                                whiskerprops=dict(color=colors[category], linewidth=1.5, linestyle='--'),
                                capprops=dict(color=colors[category], linewidth=1.5),
                                flierprops=dict(marker='o', markerfacecolor='white', 
                                                markeredgecolor=colors[category], markersize=3, alpha=0.5))
            
            # Add median annotation for OFF
            # median_off = np.median(off_data) * 1000  # Convert to ms
            # ax.text(np.median(off_data), y_pos - 0.08 - box_height * 0.25,
            #         f'{int(median_off)}', ha='center', va='top', fontsize=8, fontweight='bold')
    
    # Store y position for labels
    y_positions.append(base_y)
    short_name = channel_short_names.get(channel, channel)
    y_labels.append(short_name)

# Calculate mean event times for reference lines
mean_cop_onset = df['cop_onset_relative'].mean()
mean_frontal_peak = df['frontal_peak_relative'].mean()

# Add vertical reference lines
ax.axvline(x=0, color='green', linewidth=2, linestyle='-', label='Green Cue', zorder=0)
ax.axvline(x=mean_cop_onset, color='red', linewidth=2, linestyle='--', label=f'CoP Onset (mean)', zorder=0)
ax.axvline(x=mean_frontal_peak, color='purple', linewidth=2, linestyle=':', label=f'Frontal Peak (mean)', zorder=0)

# Set y-axis
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')

# Set x-axis
ax.set_xlabel('Time relative to Green Cue (s)', fontsize=12, fontweight='bold')
ax.set_xlim(-0.5, 2.0)

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='-')
ax.set_axisbelow(True)

# Title
ax.set_title('EMG On/Off Detection Times: YA vs OA\n(Relative to Green Cue)', 
                fontsize=14, fontweight='bold')

# Create custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=colors['YA'], edgecolor=colors['YA'], alpha=0.7, label='YA - Onset'),
    Patch(facecolor='white', edgecolor=colors['YA'], linestyle='--', label='YA - Offset'),
    Patch(facecolor=colors['OA'], edgecolor=colors['OA'], alpha=0.7, label='OA - Onset'),
    Patch(facecolor='white', edgecolor=colors['OA'], linestyle='--', label='OA - Offset'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='-', label='Green Cue'),
    Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='CoP Onset (mean)'),
    Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Frontal Peak (mean)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{SAVE_DIR}/emg_on_off_boxplots_YA_vs_OA.png', dpi=300, bbox_inches='tight')
plt.show()

#%% EMG On/Off Horizontal Boxplots - Success vs Failure comparison within YA and OA (Two columns)
csv_path='emg_on_off_green_absolute.csv'

# Load data
df = pd.read_csv(csv_path)
df = df[(df['reaction_time'] > 0) & (df['reaction_time'] < 1000)]
# Calculate times relative to green signal
df['on_relative'] = df['on'] - df['green']
df['off_relative'] = df['off'] - df['green']
df['cop_onset_relative'] = df['cop_onset'] - df['green']
df['frontal_peak_relative'] = df['frontal_peak'] - df['green']

# Get unique EMG channels and sort them
emg_channels = df['emg_channel'].unique()
emg_channels = sorted(emg_channels)

# Create short names for EMG channels
channel_short_names = {
    '01_ri_soleus': 'R Sol',
    '02_ri_gastroc_med': 'R GMed',
    '03_ri_tib_ant': 'R TA',
    '06_le_tib_ant': 'L TA',
    '07_le_soleus': 'L Sol',
    '08_le_gastroc_med': 'L GMed'
}

# Set up the figure with two columns
fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

# Colors for Success and Failure
colors = {'Success': '#2ecc71', 'Failure': '#e74c3c'}  # Green for Success, Red for Failure

# Position parameters
n_channels = len(emg_channels)
channel_height = 1.0  # Height allocated per channel
box_height = 0.35     # Height of each box
gap_between_categories = 0.05  # Gap between Success and Failure boxes

# Calculate mean event times for reference lines (for each category)
categories = ['YA', 'OA']
category_titles = {'YA': 'Young Adults (YA)', 'OA': 'Older Adults (OA)'}

for ax_idx, category in enumerate(categories):
    ax = axes[ax_idx]
    df_cat = df[df['category'] == category]
    
    y_positions = []
    y_labels = []
    
    for i, channel in enumerate(emg_channels):
        channel_data = df_cat[df_cat['emg_channel'] == channel]
        base_y = (n_channels - 1 - i) * channel_height  # Reverse order so first channel is at top
        
        for success_idx, success_val in enumerate([True, False]):
            success_label = 'Success' if success_val else 'Failure'
            success_data = channel_data[channel_data['success'] == success_val]
            
            # Calculate y position for this success condition
            if success_idx == 0:  # Success - upper position
                y_pos = base_y + gap_between_categories + box_height/2
            else:  # Failure - lower position
                y_pos = base_y - gap_between_categories - box_height/2
            
            # Get on and off data
            on_data = success_data['on_relative'].dropna()
            off_data = success_data['off_relative'].dropna()
            
            # Plot ON boxplot (filled box)
            if len(on_data) > 0:
                bp_on = ax.boxplot([on_data], positions=[y_pos + 0.08], vert=False, 
                                    widths=box_height * 0.4, patch_artist=True,
                                    boxprops=dict(facecolor=colors[success_label], alpha=0.7, linewidth=1.5),
                                    medianprops=dict(color='black', linewidth=2),
                                    whiskerprops=dict(color=colors[success_label], linewidth=1.5),
                                    capprops=dict(color=colors[success_label], linewidth=1.5),
                                    flierprops=dict(marker='o', markerfacecolor=colors[success_label], 
                                                    markersize=3, alpha=0.5))
            
            # Plot OFF boxplot (unfilled box with dashed lines)
            if len(off_data) > 0:
                bp_off = ax.boxplot([off_data], positions=[y_pos - 0.08], vert=False,
                                    widths=box_height * 0.4, patch_artist=True,
                                    boxprops=dict(facecolor='white', edgecolor=colors[success_label], 
                                                    linewidth=1.5, linestyle='--'),
                                    medianprops=dict(color='black', linewidth=2),
                                    whiskerprops=dict(color=colors[success_label], linewidth=1.5, linestyle='--'),
                                    capprops=dict(color=colors[success_label], linewidth=1.5),
                                    flierprops=dict(marker='o', markerfacecolor='white', 
                                                    markeredgecolor=colors[success_label], markersize=3, alpha=0.5))
        
        # Store y position for labels
        y_positions.append(base_y)
        short_name = channel_short_names.get(channel, channel)
        y_labels.append(short_name)
    
    # Calculate mean event times for reference lines (within this category)
    mean_cop_onset = df_cat['cop_onset_relative'].mean()
    mean_frontal_peak = df_cat['frontal_peak_relative'].mean()
    
    # Add vertical reference lines
    ax.axvline(x=0, color='green', linewidth=2, linestyle='-', label='Green Cue', zorder=0)
    ax.axvline(x=mean_cop_onset, color='red', linewidth=2, linestyle='--', label=f'CoP Onset (mean)', zorder=0)
    ax.axvline(x=mean_frontal_peak, color='purple', linewidth=2, linestyle=':', label=f'Frontal Peak (mean)', zorder=0)

    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
    # Set x-axis
    ax.set_xlabel('Time relative to Green Cue (s)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 2.0)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Title for each subplot
    ax.set_title(f'{category_titles[category]}\nSuccess vs Failure', fontsize=14, fontweight='bold')

# Create custom legend (only on the right subplot to avoid duplication)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=colors['Success'], edgecolor=colors['Success'], alpha=0.7, label='Success - Onset'),
    Patch(facecolor='white', edgecolor=colors['Success'], linestyle='--', label='Success - Offset'),
    Patch(facecolor=colors['Failure'], edgecolor=colors['Failure'], alpha=0.7, label='Failure - Onset'),
    Patch(facecolor='white', edgecolor=colors['Failure'], linestyle='--', label='Failure - Offset'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='Green Cue'),
    Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='CoP Onset (mean)'),
    Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Frontal Peak (mean)')
]
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)

# Main title
fig.suptitle('EMG On/Off Detection Times: Success vs Failure\n(Relative to Green Cue)', 
             fontsize=16, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{SAVE_DIR}/emg_on_off_boxplots_success_vs_failure_by_group.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
