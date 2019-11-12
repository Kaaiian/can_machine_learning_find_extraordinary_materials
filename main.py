import numpy as np
import pandas as pd
import os

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, KFold

from models.model import Model
from models.metrics import Meter
from models.visualize import make_figure
from data.aflow_data import AflowData

# %%


def get_models():
    # regression grid search parameters
    nnr_grid_params = {'n_neighbors': [1]}
    ridge_grid_params = {'alpha': np.logspace(-5, 2, 5)}
    svr_grid_params = {'C': np.logspace(2, 4, 5),
                       'gamma': np.logspace(-3, 1, 5)}
    # classification grid search parameters
    nnc_grid_params = {'n_neighbors': [1]}
    logreg_grid_params = {'solver': ['lbfgs'], 'C': np.logspace(-1, 4, 5)}
    svc_grid_params = {'C': np.logspace(-1, 4, 5),
                       'gamma': np.logspace(-2, 2, 5)}

    models = {}
    # regression models
    models['ridge'] = Model(Ridge, ridge_grid_params)
#    models['svr'] = Model(SVR, svr_grid_params)
    # classification models
    models['logreg'] = Model(LogisticRegression,
                             logreg_grid_params,
                             classification=True)
#    models['svc'] = Model(SVC,
#                          svc_grid_params,
#                          classification=True)
    # dumb models
    models['nnr'] = Model(KNeighborsRegressor, nnr_grid_params)
    models['nnc'] = Model(KNeighborsClassifier,
                          nnc_grid_params,
                          classification=True)
    return models


def eval_model(prop,
               gap,
               models,
               data,
               model_type,
               holdout_elem,
               holdout_struct,
               folder):
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    X_train_scaled, X_test_scaled = data[0:2]
    y_train, y_test = data[2:4]
    y_train_labeled, y_test_labeled = data[4:6]
    formula_train, formula_test = data[6:8]
    train_threshold_x, test_threshold_x = data[8:10]
    structure_train, structure_test = data[13:15]
    holdout_struct = data[15]
    gap_size = data[12]
    if model_type == 'ridge_density':
        model = models['ridge']
    else:
        model = models[model_type]
    if model.classification:
        model.fit(X_train_scaled, y_train_labeled)
        y_train_pred = cross_val_predict(model.model,
                                         X_train_scaled,
                                         y_train_labeled,
                                         cv=cv,
                                         method='predict_proba')
        y_train_pred = [probability[1] for probability in y_train_pred]
        y_train_pred = pd.Series(y_train_pred)
    else:
        model.fit(X_train_scaled, y_train)
        y_train_pred = cross_val_predict(model.model,
                                         X_train_scaled,
                                         y_train,
                                         cv=cv)
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred_prob = model.predict_proba(X_test_scaled)

    model.optimize_threshold(y_train_labeled, y_train_pred)

    # save csv files
    if holdout_elem is None and holdout_struct is None:
        csv_path = 'pred_vs_act_data/'+prop+'/'+str(gap)+'/'
        os.makedirs(csv_path, exist_ok=True)
        df_csv = pd.DataFrame(y_test)
        df_csv['predicted'] = y_test_pred
        df_csv.to_csv(csv_path+model_type+'_test.csv')

        df_csv = pd.DataFrame(y_train)
        df_csv['predicted'] = y_train_pred
        df_csv.to_csv(csv_path+model_type+'_train.csv')

    make_figure(model.threshold,
                y_test,
                y_test_pred,
                formula_test,
                gap_size=gap_size,
                test_threshold_x=test_threshold_x,
                prop=prop,
                gap=gap,
                model_type=model_type,
                classification=model.classification,
                holdout_elem=holdout_elem,
                structure=structure_test,
                holdout_struct=holdout_struct,
                folder=folder)

    display_train = False
    display_train = True
    if display_train:
        make_figure(model.threshold,
                    y_train,
                    y_train_pred,
                    formula_train,
                    gap_size=0,
                    test_threshold_x=train_threshold_x,
                    prop=prop,
                    gap=gap,
                    model_type=model_type+'train',
                    classification=model.classification,
                    holdout_elem=holdout_elem,
                    structure=structure_train,
                    holdout_struct=holdout_struct,
                    folder=folder)

    output = [model.threshold,
              y_test,
              y_test_labeled,
              y_test_pred,
              y_test_pred_prob]

    return output


def main(holdout_elem=None, holdout_structure=None, folder='figures'):
    aflow_data = AflowData()
    props = ['ael_bulk_modulus_vrh',
             'ael_debye_temperature',
             'ael_shear_modulus_vrh',
             'agl_thermal_conductivity_300K',
             'agl_thermal_expansion_300K',
             'Egap']
    gaps = [0,
            4,
            8,
            12]
    model_types = list(get_models().keys())
    meter = Meter(props, gaps, model_types)
    for prop in props:
        for gap in gaps:
            models = get_models()
            data = aflow_data.get_split(prop,
                                        elem_prop='oliynyk',
                                        gap=gap,
                                        seed_num=10,
                                        holdout_elem=holdout_elem,
                                        holdout_only=False,
                                        holdout_structure=holdout_structure
                                        )
            for model_type in model_types:
                output = eval_model(prop,
                                    gap,
                                    models,
                                    data,
                                    model_type,
                                    aflow_data.holdout_elem,
                                    holdout_structure,
                                    folder=folder
                                    )
                meter.update(prop, gap, model_type, output)

            # compare bulk prediction to density rule-of-thumb
            if prop == 'ael_bulk_modulus_vrh':
                meter.model_types = model_types + ['ridge_density']
                data = aflow_data.get_split(prop,
                                            elem_prop='oliynyk',
                                            gap=gap,
                                            seed_num=1,
                                            holdout_elem=holdout_elem,
                                            holdout_only=False,
                                            density_feat=True,
                                            holdout_structure=holdout_structure
                                            )
                output = eval_model(prop,
                                    gap,
                                    models,
                                    data,
                                    'ridge_density',
                                    aflow_data.holdout_elem,
                                    holdout_structure,
                                    folder=folder)
                meter.update(prop, gap, 'ridge_density', output)
                model_types = list(get_models().keys())
    return meter


if __name__ == '__main__':
    # run with normal train-test split
    holdout_elem = None
    holdout_structure = None
    save_dir = 'figures_default'
    meter = main(holdout_elem, holdout_structure, folder=save_dir)
    meter.metrics()
    meter.plot_curve(curve='roc', folder=save_dir)
    meter.plot_curve(curve='pr', folder=save_dir)
    meter.save(save_dir)

    # remove most common element in 'extraordinary data' from training data
    holdout_elem = 0
    holdout_structure = None
    save_dir = 'figures_missing_elem'
    meter = main(holdout_elem, holdout_structure, folder=save_dir)
    meter.metrics()
    meter.plot_curve(curve='roc', folder=save_dir)
    meter.plot_curve(curve='pr', folder=save_dir)
    meter.save(save_dir)

    # remove most common structure in 'extraordinary data' from training data
    holdout_elem = None
    holdout_structure = 0
    save_dir = 'figures_missing_struct'
    meter = main(holdout_elem, holdout_structure, folder=save_dir)
    meter.metrics()
    meter.plot_curve(curve='roc', folder=save_dir)
    meter.plot_curve(curve='pr', folder=save_dir)
    meter.save(save_dir)



