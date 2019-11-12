import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from data.aflow_data import AflowData

plt.rcParams.update({'font.size': 22})

fig_dir = 'pred_vs_act_data/'

props = os.listdir(fig_dir)
props = [prop for prop in props if 'py' not in prop]
gaps = os.listdir(fig_dir + props[0])
gaps = sorted(gaps, key=lambda x: float(x))

prop = 'ael_bulk_modulus_vrh'
#prop = 'ael_shear_modulus_vrh'
gap = '0'

df = pd.read_csv(fig_dir+prop+'/'+gap+'/'+'ridge_test.csv', index_col=0)
df_train = pd.read_csv(fig_dir+prop+'/'+gap+'/'+'ridge_train.csv', index_col=0)

df_prop = pd.concat([df, df_train])
df_prop = df_prop.sort_values('target')
cutoff = df_prop.iloc[-int(0.01 * len(df_prop))]['target']

idx_low = df[df['target'] > cutoff]['predicted'].idxmin()
df_low = df.loc[idx_low]
plt.plot(df['target'], df['predicted'], 'o')
plt.plot(df_low['target'], df_low['predicted'], 'o')
plt.title(prop + gap)

aflow_data = AflowData()
props = [prop]
gaps = [0]

for prop in props:
    for gap in gaps:
        data = aflow_data.get_split(prop,
                                    elem_prop='oliynyk',
                                    gap=gap,
                                    seed_num=10,
                                    holdout_elem=None,
                                    holdout_only=False,
                                    holdout_structure=None
                                    )

X_train_scaled, X_test_scaled = data[0:2]
y_train, y_test = data[2:4]
y_train_labeled, y_test_labeled = data[4:6]
formula_train, formula_test = data[6:8]
train_threshold_x, test_threshold_x = data[8:10]
structure_train, structure_test = data[13:15]
holdout_struct = data[15]
gap_size = data[12]

X_bad_pred = X_test_scaled.loc[idx_low, :]
df_distances = pd.DataFrame(index=X_train_scaled.index)
for i in X_train_scaled.index:
    vec_diff = X_train_scaled.loc[i, :].values - X_bad_pred.values
    distance = np.linalg.norm(vec_diff)
    df_distances.loc[i, 0] = distance


n = 25
dist = df_distances.sort_values(0)[:n]
closest = df_distances.sort_values(0).index[:n]
plt.plot(df['target'], df['predicted'], 'o')
plt.plot(df_low['target'], df_low['predicted'], 'o')
#plt.plot(dist*10000-18000, y_train.loc[closest], 'r*', markersize=13)
plt.plot(df_train.loc[closest, 'target'], df_train.loc[closest, 'predicted'], 'r*', markersize=13)


df_aflow = pd.read_csv('data/master_icsd.csv')
matched = df_aflow[df_aflow['formula'].isin(formula_train.loc[closest])]
lowest = df_aflow[df_aflow['formula'] == formula_test.loc[idx_low]]

matched_struct = matched[['formula', 'natoms', 'spacegroup_relax', prop]].dropna()
lowest_struct = lowest[['formula', 'natoms', 'spacegroup_relax', prop]].dropna()

print(matched_struct,'\n\n', lowest_struct)
