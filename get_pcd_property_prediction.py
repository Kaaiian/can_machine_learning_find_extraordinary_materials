import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from data.cbfv.composition import _fractional_composition as frac_comp
from data.cbfv.composition import generate_features
from data.aflow_data import AflowData

plt.rcParams.update({'font.size': 12})


prop = 'ael_bulk_modulus_vrh'
aflow_data = AflowData()
data = aflow_data.get_split(prop,
                            elem_prop='oliynyk',
                            gap=0,
                            seed_num=10,
                            holdout_elem=None,
                            holdout_only=False,
                            holdout_structure=None)

X_train_scaled, X_test_scaled = data[0:2]
y_train, y_test = data[2:4]
y_train_labeled, y_test_labeled = data[4:6]
formula_train, formula_test = data[6:8]
train_threshold_x, test_threshold_x = data[8:10]
scaler, normalizer = data[10:12]
structure_train, structure_test = data[13:15]


def random_guess():
    p = []
    r = []
    f = []
    for i in range(20):
        guess = np.random.permutation(y_test_labeled.values)
        metr = metrics.precision_recall_fscore_support(y_test_labeled.values,
                                                       guess)
        precision, recall, fscore, support = metr
        p.append(precision[1])
        r.append(recall[1])
        f.append(fscore[1])
    print('Precision', np.mean(p), 'recall', np.mean(r), 'f', np.mean(f))


def get_non_DFT_compounds(formula_all, extraordinary_formulae):
    in_train = []
    out_of_train = []
    fractional_formula_all = [frac_comp(formula) for formula in formula_all]
    for formula in extraordinary_formulae:
        fractional_formula = frac_comp(formula)
        if fractional_formula not in fractional_formula_all:
            out_of_train.append(formula)
        else:
            in_train.append(formula)
    return in_train, out_of_train


def read_in_elpasolite():
    df_elp = pd.DataFrame()
    formula_elp = pd.read_csv('data/elpasolites.csv', squeeze=True)
    target_elp = [0] * formula_elp.shape[0]
    df_elp['formula'] = formula_elp.values
    df_elp['target'] = target_elp
    feat_info = generate_features(df_elp, elem_prop='oliynyk')
    X_elp, y_elp, formula_elp, skipped_elp = feat_info
    return X_elp, y_elp, formula_elp, skipped_elp


def read_in_pcd():
    df_pcd = pd.DataFrame()
    formula_pcd = pd.read_csv('data/PCD_valid_formulae.csv', squeeze=True)
    target_pcd = [0] * formula_pcd.shape[0]
    df_pcd['formula'] = formula_pcd.values
    df_pcd['target'] = target_pcd
    feat_info = generate_features(df_pcd, elem_prop='oliynyk')
    X_pcd, y_pcd, formula_pcd, skipped_pcd = feat_info
    return X_pcd, y_pcd, formula_pcd, skipped_pcd


def get_model(X, y, formula):
    X_scaled = normalizer.transform(scaler.transform(X))
    X_all_scaled = pd.concat([X_train_scaled, X_test_scaled],
                             ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    formula_all = pd.concat([formula_train, formula_test], ignore_index=True)
    y_all_labeled = []
    percent = 0.01
    for value in y_all:
        if value > y_all.sort_values().iloc[-int(y_all.shape[0]*percent)]:
            y_all_labeled.append(1)
        else:
            y_all_labeled.append(0)

    y_all_labeled = pd.Series(y_all_labeled)
    exceptional_formula = formula_all[y_all_labeled == 1]
    print(y_all_labeled.value_counts())

    best_params_lr = {'C': 31.622776601683793, 'class_weight': {0: 1, 1: 1}}
    lr = LogisticRegression(**best_params_lr, solver='lbfgs', random_state=1)
    lr.fit(X_all_scaled, y_all_labeled)
    return lr, X_scaled, [exceptional_formula, formula_all]


def get_pred(lr, formula_all, formula, threshold=0.5, top_N=None):

    y_prob_lr = lr.predict_proba(X_scaled)
    extraordinary_formulae_lr = []

    sorted_probs = pd.DataFrame(y_prob_lr,
                                index=formula)[1].sort_values()

    for prob, formula in zip(y_prob_lr, formula):
        if prob[1] >= threshold:
            extraordinary_formulae_lr.append(formula)
    print('# above threshold:', len(extraordinary_formulae_lr))
    if top_N is not None:
        extraordinary_formulae_lr = sorted_probs.iloc[-top_N:].index.tolist()

    in_train, out_of_train = get_non_DFT_compounds(formula_all,
                                                   extraordinary_formulae_lr)

    overlap = []
    for lr in out_of_train:
        overlap.append(lr)
    print('# of predicted compounds found in training set:', len(in_train))
    print('# of predicted compounds:', len(out_of_train))

    if len(out_of_train) < 225:
        print('\ncompounds of interest:', out_of_train)
    return in_train, out_of_train


def get_elem_hist(formula_list):
    fractions = [frac_comp(formula) for formula in formula_list]

    keys = []
    for frac in fractions:
        keys += list(frac.keys())

    elem_counter = {}
    for key in set(keys):
        elem_counter[key] = 0

    for frac in fractions:
        for key in frac.keys():
            elem_counter[key] += 1

    df_elem_counter = pd.Series(elem_counter)
    df_elem_counter.sort_values(inplace=True, ascending=False)
    df_elem_counter = df_elem_counter.iloc[:10]
    return df_elem_counter


# %%

out_pcd = read_in_pcd()
X_pcd, y_pcd, formula_pcd, skipped_pcd = out_pcd
X, y, formula = X_pcd, y_pcd, formula_pcd
lr, X_scaled, formula_data = get_model(X, y, formula)
exceptional_formula, formula_all = formula_data
in_train, out_of_train = get_pred(lr, formula_all, formula, threshold=0.8)

if len(in_train + out_of_train) == 0:
    print('\n*************************************************************' +
          '\nthere are no compounds with a probability above the threshold' +
          '\n*************************************************************')

else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    v1 = get_elem_hist(exceptional_formula)
    v2 = get_elem_hist(in_train + out_of_train)
    ax1.bar(v1.index, v1/sum(v1), edgecolor='k',
            color='#abdda6', label='AFLOW data, top 1%')
    for x, y, s in zip(v1.index, v1/sum(v1), v1.astype(str).tolist()):
        ax1.text(x, y+0.013, s, horizontalalignment='center',
                 verticalalignment='center')
    ax2.bar(v2.index, v2/sum(v2), edgecolor='k',
            color='#fdae65', label='PCD Screened')
    for x, y, s in zip(v2.index, v2/sum(v2), v2.astype(str).tolist()):
        ax2.text(x, y+0.013, s, horizontalalignment='center',
                 verticalalignment='center')
    ax1.tick_params(top=True, right=True, direction='in', length=6)
    ax2.tick_params(top=True, right=True, direction='in', length=6)
    ax1.set_ylim(0, .4)
    ax2.set_ylim(0, .4)
    ax1.legend()
    ax2.legend()
    plt.savefig('figures_post/PCD_predictions.png', dpi=300, bbox_inches='tight')
    plt.draw()
    plt.pause(0.001)
    plt.close()


# %%

out_elp = read_in_elpasolite()
X_elpasolite, y_elpasolite, formula_elpasolite, skipped_elpasolite = out_elp
X, y, formula = X_elpasolite, y_elpasolite, formula_elpasolite
lr, X_scaled, formula_data = get_model(X, y, formula)
exceptional_formula, formula_all = formula_data
in_train, out_of_train = get_pred(lr, formula_all, formula, threshold=0.5)

if len(in_train + out_of_train) == 0:
    print('\n*************************************************************' +
          '\nthere are no compounds with a probability above the threshold' +
          '\n*************************************************************')

else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    v1 = get_elem_hist(exceptional_formula)
    v2 = get_elem_hist(in_train + out_of_train)
    ax1.bar(v1.index, v1/sum(v1), edgecolor='k',
            color='#abdda6', label='Labeled "Extraordinary" for Training')
    for x, y, s in zip(v1.index, v1/sum(v1), v1.astype(str).tolist()):
        ax1.text(x, y+0.013, s, horizontalalignment='center',
                 verticalalignment='center')
    ax2.bar(v2.index, v2/sum(v2), edgecolor='k',
            color='#fdae65', label='Predicted "Extraordinary", PCD Screening')
    for x, y, s in zip(v2.index, v2/sum(v2), v2.astype(str).tolist()):
        ax2.text(x, y+0.013, s, horizontalalignment='center',
                 verticalalignment='center')
    ax1.tick_params(top=True, right=True, direction='in', length=6)
    ax2.tick_params(top=True, right=True, direction='in', length=6)
    ax1.set_ylim(0, .4)
    ax2.set_ylim(0, .4)
    ax1.legend()
    ax2.legend()
    plt.savefig('figures_post/Elpasolites_predictions.png',
                dpi=300,
                bbox_inches='tight')
    plt.draw()
    plt.pause(0.001)
    plt.close()
