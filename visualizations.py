#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

#%%
df = pd.read_csv('emg_analysis_results.csv')
cols = df.select_dtypes('number').columns.drop('subject_id')  # limits to a (float), b (int) and e (timedelta)
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
plt.savefig('./plots/emg_amplitude_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/emg_amplitude_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/emg_amplitude_OA_vs_YA.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/iemg_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/iemg_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/iemg_OA_vs_YA.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/average_peaks_YA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/average_peaks_OA_success_vs_non_success.png', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/average_peaks_OA_vs_YA.png', dpi=300, bbox_inches='tight')
# plt.show()


# %%
