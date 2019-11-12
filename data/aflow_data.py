import pandas as pd
import numpy as np
from data.cbfv.composition import generate_features, _fractional_composition
from sklearn.preprocessing import StandardScaler, Normalizer


def get_elem_hist(formula_list):
    fractions = [_fractional_composition(formula) for formula in formula_list]

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


class AflowData():
    def __init__(self):
        df_all = pd.read_csv('data/master_icsd.csv')
#        df_all = pd.read_csv('master_icsd.csv')
        num = df_all['cif_id'].str.split('_ICSD_').str[1]
        num = num.str.split('.cif').str[0].str.extract(r'(\d+)').astype(int)
        df_all['icsd_number'] = num
        df_all = df_all.sort_values('icsd_number')
        df_all = df_all.drop_duplicates('formula', keep='last')
        df_all.index = df_all['formula']
        self.df_all = df_all

    def get_split(self,
                  prop,
                  elem_prop='oliynyk',
                  gap=0,
                  seed_num=1,
                  holdout_elem=None,
                  holdout_only=False,
                  density_feat=False,
                  holdout_structure=None):
        df = self.df_all.loc[:, ['formula', prop]]
        log_props = ['ael_debye_temperature',
                     'ael_shear_modulus_vrh',
                     'agl_thermal_conductivity_300K',
                     'agl_thermal_expansion_300K']
        if prop in log_props:
            df[prop] = np.log(df[prop])
        if prop == 'Egap':
            df = df[df['Egap'] != 0]
        if prop == 'energy_atom':
            df = df[df['energy_atom'] < 10]

        df = df.loc[:, ['formula', prop]].dropna()

        df.reset_index(drop=True, inplace=True)
        # rename columns for use with vf.generate_features()
        df.columns = ['formula', 'target']
        # get composition-based feature vectors (CBFV)
        X, y, formulae, skipped = generate_features(df,
                                                    elem_prop=elem_prop)
        if density_feat:
            df_dens = self.df_all.loc[formulae, :]
            X = pd.DataFrame(df_dens['density'])
            y = df_dens[prop]
            formulae = df_dens['formula']

        df_int = self.df_all.loc[formulae, :].copy()
        struc_id = df_int['spacegroup_relax'].astype(str) + '-' + df_int['natoms'].astype(str)
        struc_id.reset_index(inplace=True, drop=True)

        # reset indices
        y.reset_index(inplace=True, drop=True)
        X.reset_index(inplace=True, drop=True)
        formulae.reset_index(inplace=True, drop=True)
        # remake the full dataframe including features and targets
        df_featurized = X.copy()
        df_featurized['formula'] = formulae
        df_featurized['target'] = y
        df_featurized['structure'] = struc_id
        # sort by the target value so the 'test' set is all extrapolation
        df_featurized.sort_values(by=['target'], inplace=True)
        # reset the index
        df_featurized.reset_index(inplace=True, drop=True)

        # remove the "extraordinary" compounds and buffer the training data
        n_extraordinary = int(df_featurized.shape[0] * 0.01)
        buffer = int(df_featurized.shape[0] * gap * 0.01)
        cut = n_extraordinary + buffer
        holdout = holdout_elem
        df_train = df_featurized.iloc[0:-cut, :]
        # set X% of the train as "ordinary" compounds for the test data
        n_test = int(df_train.shape[0] * 0.15)
        df_test_false = df_train.sample(n_test, random_state=seed_num)
        # remove these compounds from the train data
        df_train = df_train[~df_train.index.isin(df_test_false.index.values)]
        # set the top 1% "extraordinary" compounds for the test data
        df_test_true = df_featurized.iloc[-n_extraordinary:, :]
        structs = df_test_true['structure'].value_counts().index.tolist()
        self.holdout_elem = None
        if holdout is not None:
            if type(holdout) is int:
                holdout = get_elem_hist(df_test_true['formula'].values.tolist()).index[0]
            self.holdout_elem = holdout
            comps = df_train['formula'].str.split(r'([A-Z][a-z]*)')
            boolean = [True if holdout in comp else False for comp in comps]
            boolean_inv = [not i for i in boolean]
            df_transfer = df_train[boolean]
            df_train = df_train[boolean_inv]
            df_test_false = pd.concat([df_test_false, df_transfer])
        holdout_struct = None
        if holdout_structure is not None:
            holdout_struct = structs[holdout_structure]
            boolean = df_train['structure'] == holdout_struct
            boolean_inv = [not i for i in boolean]
            df_transfer = df_train[boolean]
            df_train = df_train[boolean_inv]
            df_test_false = pd.concat([df_test_false, df_transfer])

        # compile the test data "ordinary" + "extraordinary"
        df_test = pd.concat([df_test_false, df_test_true])

        # split the train and test data into features X, and target values y
        X_train = df_train.iloc[:, :-3]
        y_train = df_train.loc[:, 'target']
        formula_train = df_train.loc[:, 'formula']
        structure_train = df_train.loc[:, 'structure']

        X_test = df_test.iloc[:, :-3]
        y_test = df_test.loc[:, 'target']
        formula_test = df_test.loc[:, 'formula']
        structure_test = df_test.loc[:, 'structure']
        # Here we convert the problem from a regression
        # to a classification problem
        y_train_label = y_train.copy()
        y_test_label = y_test.copy()

        # label extraordinary compounds in train and test set
        n_test_extraordinary = df_test_true.shape[0]
        n_test_ordinary = (y_test.shape[0] - n_test_extraordinary)
        test_ratio = df_test_true.shape[0] / df_test_false.shape[0]
        n_train_extraordinary = int(test_ratio * df_train.shape[0])
        n_train_ordinary = (y_train.shape[0] - n_train_extraordinary)
        y_train_label.iloc[:-n_train_extraordinary] = [0] * n_train_ordinary
        y_train_label.iloc[-n_train_extraordinary:] = [1] * n_train_extraordinary
        y_test_label.iloc[:-n_test_extraordinary] = [0] * n_test_ordinary
        y_test_label.iloc[-n_test_extraordinary:] = [1] * n_test_extraordinary
        # get thresholds
        train_threshold_x = y_train.iloc[-y_train_label.sum().astype(int)]
        test_threshold_x = y_test.iloc[-y_test_label.sum().astype(int)]
        gap_size = test_threshold_x - y_train.max()

        # scale each column of data to have a mean of 0 and a variance of 1
        scaler = StandardScaler()
        # normalize each row in the data
        normalizer = Normalizer()
        if density_feat:
            class Nada():
                def fit_transform(self, x):
                    return x
                def transform(self, x):
                    return x
            scaler = Nada()
            normalizer = Nada()

        # fit and transform the training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(normalizer.fit_transform(X_train_scaled),
                                      columns=X_train.columns.values,
                                      index=X_train.index.values)
        # transform the test data based on training data fit
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(normalizer.transform(X_test_scaled),
                                     columns=X_test.columns.values,
                                     index=X_test.index.values)

        if holdout_only:
            comps = formula_test.str.split(r'([A-Z][a-z]*)')
            boolean = [True if holdout in comp else False for comp in comps]
            X_test_scaled = X_test_scaled[boolean]
            y_test = y_test[boolean]
            y_test_label = y_test_label[boolean]
            formula_test = formula_test[boolean]

        data = [X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                y_train_label,
                y_test_label,
                formula_train,
                formula_test,
                train_threshold_x,
                test_threshold_x,
                scaler,
                normalizer,
                gap_size,
                structure_train,
                structure_test,
                holdout_struct]

        return data


class PredictionData():
    def __init__(self, data_path):
        df_all = pd.read_csv(data_path)
        df_all['icsd_number'] = num
        df_all = df_all.sort_values('icsd_number')
        df_all = df_all.drop_duplicates('formula')
        self.df_all = df_all


if __name__ == '__main__':
    prop = 'ael_bulk_modulus_vrh'
    gap = 5
    aflow_data = AflowData()
    data = aflow_data. get_split(prop,
                   elem_prop='oliynyk',
                   gap=0,
                   seed_num=1)

#plt.figure(figsize=(7,7))
#plt.plot(y_test, X_test_scaled, 'o')
#plt.plot(y_train, X_train_scaled, 'o')
