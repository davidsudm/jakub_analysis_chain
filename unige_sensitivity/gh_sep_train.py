#!/usr/bin/env python

"""
Usage:
  ./gh_sep_train.py --optimize --train_proton=train_proton.h5 --train_gamma=train_gamma.h5 --test_proton=test_proton.h5 --test_gamma=test_gamma.h5
  ./gh_sep_train.py --train_proton=train_proton.h5 --train_gamma=train_gamma.h5 --test_proton=test_proton.h5 --test_gamma=test_gamma.h5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.reco import utils
import astropy.units as u
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from scipy import optimize
#import mglearn

dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'


parser = argparse.ArgumentParser()

parser.add_argument('--optimize', '-o', dest='optimize', action='store_const', default=False, const=True)

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )
parser.add_argument('--train_gamma', '-traing', action='store', type=str,
                    dest='train_gamma',
                    default=None
                    )

parser.add_argument('--test_gamma', '-testg', action='store', type=str,
                    dest='test_gamma',
                    default=None
                    )

parser.add_argument('--train_proton', '-trainp', action='store', type=str,
                    dest='train_proton',
                    default=None
                    )

parser.add_argument('--test_proton', '-testp', action='store', type=str,
                    dest='test_proton',
                    default=None
                    )

parser.add_argument('--telescope', '-t', action='store', type=int,
                    dest='telescope',
                    default=1
                    )

parser.add_argument('--cam_key', '-k', action='store', type=str,
                    dest='dl1_params_camera_key',
                    default=dl1_params_lstcam_key
                    )

parser.add_argument('--intensity_cut', '-i', action='store', type=float,
                    dest='intensity_cut',
                    default=300
                    )

args = parser.parse_args()

def set_figures():

    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['legend.numpoints'] = 1 #aby se u errorbaru a bodovych grafu nezobrazoval kazdy bod 2x
    mpl.rcParams['lines.markersize'] = 15

def quality_cuts(data,
                 intensity_cut=300,
                 leak_cut=0.1,
                 islands_cut=2,
                 ):
    #mask = (10**data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    data_masked = data[mask]
    return data_masked

def data_prepare(filename, key=dl1_params_lstcam_key, filters=None, telescope=1, quality=True, intensity_cut=300):

    # nacteni pouze sloupcu s parametry do pandas dataframe
    param = pd.read_hdf(filename, key=key)
    param = utils.filter_events(param, filters=events_filters)

    # zakladni statistika dat
    print('Filename:', filename)
    print('Size of dataset:', param.shape[0])

    # vyber jednoho ze simulovanych telescopu
    param = param.where(param.tel_id == telescope)
    param = param.dropna()
    print('Size of dataset (1 tel):', param.shape[0])

    # Application of selection cuts
    if quality:
        param = quality_cuts(param, intensity_cut=intensity_cut)
        param = param.dropna()
        print('Size of dataset after selection cuts:', param.shape[0])
    else: print('No quality cuts applied')

    param = shuffle(param).reset_index(drop=True)
    return param

def plot_feature_importances(X, model):
    plt.figure()
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()

def data_equalization(data_gamma, data_proton, method='down-sample'):

    if method == 'up-sample':
        if len(data_gamma) > len(data_proton):
            data_proton = resample(data_proton, replace=True, n_samples=len(data_gamma), random_state=42)
        else:
            data_gamma = resample(data_gamma, replace=True, n_samples=len(data_proton), random_state=42)
    elif method == 'down-sample':
        if len(data_gamma) > len(data_proton):
            data_gamma = resample(data_gamma, replace=False, n_samples=len(data_proton), random_state=42)
        else:
            data_proton = resample(data_proton, replace=False, n_samples=len(data_gamma), random_state=42)
    print('Gammas after equalization: ', len(data_gamma))
    print('Protons after equalization: ', len(data_proton))
    return data_gamma, data_proton


def plot_roc_curve(y, y_pred_prob, filename='sep_tpr_fpr.png', figsize=(6, 5)):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='random forest')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    return fpr, tpr, thresholds


def plot_roc_curves(y, y_pred_prob, val_filtered, val_name='', nfilter=4, log_bins=False, filename='roc_curves.png', figsize=(8, 6)):
    val_min = np.min(val_filtered)
    val_max = np.max(val_filtered)
    val_max += 0.01*(val_max-val_min)/nfilter # to include the highest value 
    if log_bins:
        bins_filter = np.logspace(np.log10(val_min), np.log10(val_max), nfilter+1)
    else:
        bins_filter = np.linspace(val_min, val_max, nfilter+1)
    fig = plt.figure(figsize=figsize)
    for b in range(nfilter):
        bin_min = bins_filter[b]
        bin_max = bins_filter[b+1]
        select = np.logical_and(bin_min <= val_filtered, val_filtered < bin_max)
        n_selected = np.sum(select)
        fpr, tpr, thresholds = metrics.roc_curve(y[select], y_pred_prob[select])
        plt.plot(fpr, tpr, label=f'{bin_min:.2g} <= {val_name} ({n_selected} events) < {bin_max:.2g}')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close(fig)



if __name__ == '__main__':

    set_figures()

    train_filename_gamma = args.train_gamma
    test_filename_gamma = args.test_gamma
    train_filename_proton = args.train_proton
    test_filename_proton = args.test_proton

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)
    events_filters = config["events_filters"]

    param_train_gamma = data_prepare(train_filename_gamma, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)
    param_test_gamma = data_prepare(test_filename_gamma, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)
    param_train_proton = data_prepare(train_filename_proton, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)
    param_test_proton = data_prepare(test_filename_proton, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)

    # adding column with 1..gamma, 0..proton flag
    param_train_gamma['target'] = np.ones(param_train_gamma['width'].size)
    param_test_gamma['target'] = np.ones(param_test_gamma['width'].size)
    param_train_proton['target'] = np.zeros(param_train_proton['width'].size)
    param_test_proton['target'] = np.zeros(param_test_proton['width'].size)

    # merge gamma and proton dataset
    param_train = pd.concat([param_train_gamma, param_train_proton])
    param_train = shuffle(param_train).reset_index(drop=True)
    param_test = pd.concat([param_test_gamma, param_test_proton])
    param_test = shuffle(param_test).reset_index(drop=True)

    features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']
    #features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis', 'phi', 'psi', 'wl'] #, 'area', 'area_log10size', 'size_lw', 'abs_skew']     # no timing


    # Split into features and target
    X_trainval = param_train[features]
    y_trainval = param_train['target']
    X_test = param_test[features]
    y_test = param_test['target']

    # split data into TRAINING / VALIDATION / TEST set
    indices = range(X_trainval.shape[0])
    train_index, validation_index = train_test_split(indices, random_state=0)
    X_train, X_validation = X_trainval.iloc[train_index], X_trainval.iloc[validation_index]
    y_train, y_validation = y_trainval.iloc[train_index], y_trainval.iloc[validation_index]

    print('X_train+validation shape: ', X_trainval.shape[0])
    print('X_test shape: ', X_test.shape[0])

    # Number of cores used
    NJOBS=20

    # Grid search of optimal parameters and cross-validation of ML model stability
    if args.optimize:

        n_cross_validations = 5
        print('X_validation shape: ', int(X_trainval.shape[0] / 5))
        print('Cross-validation and grid search: Random forest regression')

        param_grid =    {'n_estimators': [10, 50, 100, 150, 200, 300],
                        'max_depth': [5, 7, 10, 15, 20, 30],
                        'min_samples_split': [2, 3, 5, 8, 10, 15],
                        'min_samples_leaf': [1, 2, 3, 4, 5]}

        print("Parameter grid:\n{}".format(param_grid))
        grid_search = GridSearchCV(RandomForestClassifier(max_features='sqrt', random_state=42, bootstrap=True, criterion='gini'), param_grid, cv=n_cross_validations, n_jobs=NJOBS, verbose=10, scoring='roc_auc')
        grid_search.fit(X_trainval, y_trainval) # _trainval is being split internaly during cross-validation
        print("RF: best AUC on TEST set: {:.2f}".format(grid_search.score(X_test, y_test)))
        print("Best parameters: {}".format(grid_search.best_params_))

        # Analyzing output of GridSearch
        results = pd.DataFrame(grid_search.cv_results_)
        print(results)

        # prediction on test set using the best model
        reg_rf = grid_search.best_estimator_
        y_comp_rf = reg_rf.predict(X_test)
        n_nodes = [t.tree_.node_count for t in reg_rf.estimators_]
        print("Number of trees    {:d}".format(len(n_nodes)))
        print("Number of nodes (mean, min, max)    {:.2f}, {:d}, {:d}".format(np.mean(n_nodes), min(n_nodes), max(n_nodes)))
        print("Training/test set AUC:    {:.4f}/{:.4f}".format(reg_rf.score(X_trainval, y_trainval), reg_rf.score(X_test, y_test)))

        filename_rf = 'separation_rf_model.joblib'
        joblib.dump(reg_rf, filename_rf)

    # CUTS OPTIMIZATION ON VALIDATION DATASET
    #########################################
    if args.optimize:
        reg_rf = grid_search.best_estimator_
    else:
        reg_rf = RandomForestClassifier(max_depth=30, min_samples_leaf=3, min_samples_split=10, n_estimators=300, max_features='sqrt', n_jobs=NJOBS, random_state=42, bootstrap=True, criterion='gini')
    reg_rf.fit(X_train, y_train)
    score_rf_test = metrics.roc_auc_score(y_validation, reg_rf.predict_proba(X_validation)[:, 1])  # AUC score is used for evaluation
    y_comp_rf = reg_rf.predict(X_validation)
    y_pred_prob = reg_rf.predict_proba(X_validation)

    # precision and recall
    ######################
    # precision = true positive / (true positive + false positive)
    # - tohle by mel byt dominantni parametr pokud jde hlavne o to potlacit protony identifikovana spatne jako gamma
    # - je to mira toho, kolik z predpovezenych gamma je opravdu gamma
    # recall = true positive / (true positive + false negative)
    # - mira toho, kolik z gamma je opravdu identifikovana jako gamma
    # - tohle by mel byt dominantni parametr, pokud nechci prijit separaci o hodne gamma eventu
    # y_pred_prob[:, 1] - pravdepodobnost, ze je event gamma
    precision, recall, thresholds = metrics.precision_recall_curve(y_validation, y_pred_prob[:, 1])

    mask = recall[:-1] > 0
    recall = recall[:-1][mask]
    thresholds = thresholds[mask]
    precision = precision[:-1][mask]

    # F1_score
    # = (1 + beta**2) * prec * recall / (beta**2 * prec + recall)
    # - defaultne beta = 1 v te build in funkci
    # - pokud beta > 1 tak to uprednostni precision, naopak recall
    ##########
    f1_score = []
    cuts = np.linspace(0, 1, 100)
    for threshold in cuts:
        mask_gamma = y_pred_prob[:, 1] > threshold
        y_pred_t = np.zeros(len(y_validation))
        y_pred_t[mask_gamma] = 1
        f1_score.append(metrics.f1_score(y_validation, y_pred_t))

    # cut_optimal - pokud je pravdepodobnost y_pred_prob[:, 1] > cut, oznaci se event jako gamma
    index = np.argmax(f1_score)
    cut_optimal = cuts[index]
    print('\nCUTS OPTIMIZATION ON VALIDATION DATASET\n')
    print('Validation: Best F1 score:                           {:.4f}'.format(f1_score[index]))
    print('Validation: Optimal threshold according to F1_score: {:.4f}'.format(cut_optimal))
    index = np.argmin(abs(cut_optimal-thresholds))
    print('Validation: Precision at given threshold:            {:.4f}'.format(precision[index]))
    print('Validation: Recall at given threshold:               {:.4f}'.format(recall[index]))

    plt.figure(figsize=(6, 5))
    plt.plot(cuts, f1_score, '-')
    plt.xlabel('threshold')
    plt.ylabel('F1 score')
    plt.tight_layout()
    plt.savefig('sep_f1_score.png', dpi=100)

    # Hadroness
    hadroness_cut = 1 - cut_optimal

    # ROC curve
    fpr, tpr, thresholds = plot_roc_curve(y_validation, y_pred_prob[:, 1], filename='sep_tpr_fpr_validation.png')
    index = np.argmin(abs(cut_optimal-thresholds))
    print('Validation: False positive rate:                     {:.4f}'.format(fpr[index]))
    print('Validation: True positive rate:                      {:.4f}'.format(tpr[index]))

    print('\nRF: best AUC on VALIDATION set:                    {:.4f}'.format(score_rf_test))

    # rebuild model for best parameters on combined _trainval set
    #############################################################
    if args.optimize:
        reg_rf = grid_search.best_estimator_
    else:
        reg_rf = RandomForestClassifier(max_depth=30, min_samples_leaf=3, min_samples_split=10, n_estimators=300, max_features='sqrt', n_jobs=NJOBS, random_state=42, bootstrap=True, criterion='gini')
    reg_rf.fit(X_trainval, y_trainval)
    y_comp_rf = reg_rf.predict(X_test)
    score_rf_test = metrics.roc_auc_score(y_test, reg_rf.predict_proba(X_test)[:, 1])  # AUC score is used for evaluation
    y_pred_prob = reg_rf.predict_proba(X_test)

    # Saving the model on disk
    if ~args.optimize:
        filename_rf = 'separation_rf_model.joblib'
        joblib.dump(reg_rf, filename_rf)

    # Print results
    print('RF: best AUC on TEST set:                            {:.4f}'.format(score_rf_test))

    print('\nPERFORMANCE ON TEST SET\n')

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob[:, 1])
    mask = recall[:-1] > 0
    recall = recall[:-1][mask]
    thresholds = thresholds[mask]
    precision = precision[:-1][mask]

    index = np.argmin(abs(cut_optimal-thresholds))
    print('Test: Precision at optimal threshold:                  {:.4f}'.format(precision[index]))
    print('Test: Recall at optimal threshold:                     {:.4f}'.format(recall[index]))

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, precision, label='precision')
    plt.plot(thresholds, recall, label='recall')
    plt.xlabel('threshold (confidence cut)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sep_thres_precision_recall.png', dpi=100)

    # Hadroness
    hadroness = y_pred_prob[:, 0]    # protoze protony znacime 0, takze pravdepodobnost tehle tridy je v 0. sloupci
    hadroness_gamma = hadroness[y_test == 1]
    hadroness_proton = hadroness[y_test == 0]

    # ROC curve
    fpr, tpr, thresholds = plot_roc_curve(y_test, y_pred_prob[:, 1], filename='sep_tpr_fpr.png')
    index = np.argmin(abs(cut_optimal-thresholds))
    print('Test: Hardoness cut:                                 {:.4f}'.format(hadroness_cut))
    print('Test: False positive rate:                           {:.4f}'.format(fpr[index]))
    print('Test: True positive rate:                            {:.4f}'.format(tpr[index]))

    fpr_cut = 0.01
    print('\nFPR = {:.2f}:'.format(fpr_cut))
    index = np.argmin(abs(fpr-fpr_cut))
    hadroness_cut_001 = 1 - thresholds[index]
    print('Test: Hardoness cut:                                 {:.4f}'.format(hadroness_cut_001))
    print('Test: Threshold for {:.2f} FPR:                      {:.4f}'.format(fpr_cut, thresholds[index]))
    print('Test: True positive rate for {:.2f} FPR:               {:.4f}'.format(fpr_cut, tpr[index]))
    fpr_cut = 0.1
    print('\nFPR = {:.2f}:'.format(fpr_cut))
    index = np.argmin(abs(fpr-fpr_cut))
    hadroness_cut_01 = 1 - thresholds[index]
    print('Test: Hardoness cut:                                 {:.4f}'.format(hadroness_cut_01))
    print('Test: Threshold for {:.2f} FPR:                      {:.4f}'.format(fpr_cut, thresholds[index]))
    print('Test: True positive rate for {:.2f} FPR:               {:.4f}'.format(fpr_cut, tpr[index]))

    # FIGURES

    plot_roc_curves(y_test, y_pred_prob[:, 1], param_test['mc_energy'], val_name=r'$E_{MC}$[TeV]', nfilter=4, filename='roc_curves_energy.png', log_bins=True)

    plot_feature_importances(X_test, reg_rf)
    plt.savefig('sep_rf_features.png', dpi=100)

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, tpr, label='true_positive_rate')
    plt.plot(thresholds, fpr, label='false_positive_rate')
    #plt.plot(thresholds, tpr / np.sqrt(fpr), label='quality factor')
    plt.xlabel('threshold (confidence cut)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sep_thres_tpr_fpr.png', dpi=100)

    plt.figure(figsize=(6, 5))
    plt.plot(precision, recall)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.tight_layout()
    plt.savefig('sep_precision_recall.png', dpi=100)

    plt.figure(figsize=(6, 5))
    plt.hist(hadroness_gamma, bins=100, label='gammas', alpha=0.8, density=True)
    plt.hist(hadroness_proton, bins=100, label='protons', alpha=0.8, density=True)
    plt.axvline(hadroness_cut, color='grey', alpha=0.8)
    plt.xlabel('hadroness')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sep_hadroness.png', dpi=100)

    plt.show()
