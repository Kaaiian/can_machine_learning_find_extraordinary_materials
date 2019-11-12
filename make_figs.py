import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 22})

fig_dir = 'figures_default/'

props = os.listdir(fig_dir)
props = [prop for prop in props if 'py' not in prop]
gaps = os.listdir(fig_dir + props[0])
gaps = sorted(gaps, key=lambda x: float(x))

line = ['-', '-', '--', '--', '-.']
color = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']


def get_df_pr_auc(prop):
    df_pr_auc = pd.DataFrame()
    for gap in gaps:
        df_metric = pd.read_csv(fig_dir+prop+'/'+gap+'/'+'metrics.csv',
                                index_col=0)
#        pr_auc.append(df_metric['pr_auc'])
#        df_metric['pr_auc'].plot(label=gap)
        df_pr_auc[float(gap)] = df_metric['pr_auc']
#        df_pr_auc[float(gap)] = df_metric['fscore']
#        df_pr_auc[float(gap)] = df_metric['recall']
    if prop == 'ael_bulk_modulus_vrh':
        df_pr_auc = df_pr_auc.loc[['ridge', 'logreg', 'nnr', 'nnc', 'ridge_density']]
        df_pr_auc.index = ['ridge', 'logreg', 'nnr', 'nnc', 'density']
#        df_pr_auc = df_pr_auc.loc[['nnr', 'nnc', 'ridge', 'logreg', 'ridge_density']]
#        df_pr_auc.index = ['nnr', 'nnc', 'ridge', 'logreg', 'density']
    else:
        df_pr_auc = df_pr_auc.loc[['ridge', 'logreg', 'nnr', 'nnc']]
    return df_pr_auc

def plot_df_pr_auc(df_pr_auc, save_dir):
    plt.figure(figsize=(7, 7))
#    df_pr_auc.T.plot(figsize=(7, 7), linewidth=2, linestyle=line[i], color=color[i])
    for i, col in enumerate(df_pr_auc.T):
        df_pr_auc.T[col].plot(linewidth=2, linestyle=line[i], color=color[i], label=col)

    plt.xticks([0, 2.5, 5, 10])
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.tick_params(direction='in',
                length=5,
                bottom=True,
                top=True,
                left=True,
                right=True)
    plt.legend(loc=1, framealpha=0.15,
               handlelength=1.5,
               labelspacing=0.1)
    plt.title(prop.replace('_', ' ').title(), fontsize=22)
    plt.savefig(save_dir+prop, dpi=300, bbox_inches='tight')




# %%
gaps_float = [float(gap) for gap in gaps]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
axes = [ax1, ax2, ax3]
#
#fig_dirs = ['figures_default/']
#save_dirs = ['figures_post/missing_struct/']

fig_dirs = ['figures_default/',
            'figures_missing_elem/',
            'figures_missing_struct/']
save_dirs = ['figures_post/gap/',
             'figures_post/missing_elem/',
             'figures_post/missing_struct/']

titles = ['all data', 'element removed', 'structure removed']

i_count = 0
for fig_dir, save_dir, ax in zip(fig_dirs, save_dirs, axes):
    os.makedirs(save_dir, exist_ok=True)
    df_list = []
    for prop in props:
        df_pr_auc = get_df_pr_auc(prop)
        df_list.append(df_pr_auc)

    mat = np.zeros(shape=(4, df_list[0].shape[-1]))
    for df in df_list:
        mat += df.iloc[:4, :].values
    mat = mat / len(df_list)
    df_avg = pd.DataFrame(mat, index = df_list[1].index, columns=df_list[1].columns)

    for i, col in enumerate(df_avg.T):
        if col == 'logreg':
            df_avg.T[col].plot(linewidth=3, linestyle=line[i], color=color[i], label='Logistic', ax=ax)
        elif col == 'ridge':
            df_avg.T[col].plot(linewidth=3, linestyle=line[i], color=color[i], label='Ridge', ax=ax)
        else:
            df_avg.T[col].plot(linewidth=3, linestyle=line[i], color=color[i], label=col, ax=ax)
    df_list.append(df_pr_auc)
    ax.set_xticks(gaps_float)
    ax.xaxis.set_ticklabels(gaps)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.tick_params(direction='in',
                length=8,
                bottom=True,
                top=True,
                left=True,
                right=True)
    ax.set_title(titles[i_count], fontsize=22)
    i_count = i_count + 1
ax2.legend(loc=0, framealpha=0.15,
           handlelength=1.3,
           labelspacing=0.1)
ax1.yaxis.set_ticklabels(['', 0.2, 0.4, 0.6, 0.8, 1])
ax2.yaxis.set_ticklabels([])
ax3.yaxis.set_ticklabels([])
ax1.set_ylabel('precision-recall AUC')
#ax1.set_ylabel('fscore')
ax2.set_xlabel('gap size (% data)')
plt.subplots_adjust(wspace=0.12)
plt.savefig('figures_post/'+'averaged.png', dpi=300, bbox_inches='tight')
