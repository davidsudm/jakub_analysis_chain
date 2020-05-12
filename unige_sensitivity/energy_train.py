#!/usr/bin/env python

"""
Usage:
  ./energy_train.py --optimize --train_file=train_file.h5 --test_file=test_file.h5
  ./energy_train.py --train_file=train_file.h5 --test_file=test_file.h5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.reco import utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from scipy import optimize
import time
#import mglearn

dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'

parser = argparse.ArgumentParser()

parser.add_argument('--optimize', '-o', dest='optimize', action='store_const', default=False, const=True)

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--train_file', '-train', action='store', type=str,
                    dest='train_file',
                    default=None
                    )

parser.add_argument('--test_file', '-test', action='store', type=str,
                    dest='test_file',
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


def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

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
    mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    #mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)     # nove MC
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

def line_point_distance(p1, c, p):

    # Distance between a straight line and a point in space.
    # p1:   numpy 1d array, reference point on the line
    # c:    numpy 1d array, direction cosines of the line
    # p:    numpy 1d array, point in space
    # From Konrad's hessio rec_tools.h, line_point_distance function
    a = np.cross((p1 - p), c)
    return np.sqrt(np.sum(a**2, axis=-1)/np.sum(c**2, axis=-1))

def impact_parameter(x_core, y_core, telpos, theta, phi):

    cx = np.sin(theta)*np.cos(phi)  # direction cosines
    cy = np.sin(theta)*np.sin(phi)  #
    cz = np.cos(theta)                          #

    direction = np.array([cx, cy, cz]).T
    impact_point = np.array([x_core, y_core, np.zeros(x_core.shape[0])]).T
    impact = line_point_distance(impact_point, direction, telpos)
    return impact


if __name__ == '__main__':

    train_filename = args.train_file
    test_filename = args.test_file

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)
    events_filters = config["events_filters"]

    param_train = data_prepare(train_filename, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)
    param_test = data_prepare(test_filename, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True, intensity_cut=args.intensity_cut)

    # MC (true) impact parameter
    telpos = np.array([param_train.tel_pos_x[0], param_train.tel_pos_y[0], param_train.tel_pos_z[0]])
    param_train['mc_impact'] = impact_parameter(param_train.mc_core_x, param_train.mc_core_y, telpos, np.pi/2. - param_train.mc_alt_tel, param_train.mc_az_tel)
    param_test['mc_impact'] = impact_parameter(param_test.mc_core_x, param_test.mc_core_y, telpos, np.pi/2. - param_test.mc_alt_tel, param_test.mc_az_tel)

    # znamenko time gradientu by nemelo hrat roli, pokud by v tom nahodou byla nejak vyspoilovana pozice zdroje, tak absolutni hodnota zajisti ze nebude
    # - overeno, vyjde to stejne
    #param_train['abs_time_gradient'] = abs(param_train['time_gradient'])
    #param_test['abs_time_gradient'] = abs(param_test['time_gradient'])

    # no MC
    features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis']
    #features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis']  # no time_gradient

    # Split into features and target
    X = param_train[features]
    target = 10**3 * param_train['mc_energy'] / param_train['intensity']    # v novych MC uz tohle nejsou log10 veliciny a E je v TeV

    # split data into TRAINING / VALIDATION
    X_trainval = X
    y_trainval = target

    X_test = param_test[features]
    y_test = 10**3 * param_test['mc_energy'] / param_test['intensity']    # v novych MC uz tohle nejsou log10 veliciny a E je v TeV

    print('X_train+validation shape: ', X_trainval.shape[0])
    print('X_test shape: ', X_test.shape[0])

    # Number of cores used
    NJOBS=20

    # Grid search of optimal parameters and cross-validation of ML model stability
    if args.optimize:

        n_cross_validations = 5
        print('X_validation shape: ', int(X_trainval.shape[0] / 5))
        print('Cross-validation and grid search: Random forest regression')

        param_grid =    {'n_estimators': [10, 50, 100, 150],
                        'max_depth': [5, 7, 10, 15, 20, 30],
                        'min_samples_split': [2, 3, 5, 8, 10],
                        'min_samples_leaf': [1, 2, 3, 4, 5]}

        print("Parameter grid:\n{}".format(param_grid))
        grid_search = GridSearchCV(RandomForestRegressor(max_features='sqrt', random_state=42, bootstrap=True, criterion='mse'), param_grid, cv=n_cross_validations, n_jobs=NJOBS, verbose=10)
        grid_search.fit(X_trainval, y_trainval) # _trainval is being split internaly during cross-validation
        print("RF: best R^2 on TEST set: {:.2f}".format(grid_search.score(X_test, y_test)))
        print("Best parameters: {}".format(grid_search.best_params_))

        # Saving the RF model on disk
        filename_rf = 'energy_rf_model.joblib'
        joblib.dump(grid_search.best_estimator_, filename_rf)

        # Analyzing output of GridSearch
        results = pd.DataFrame(grid_search.cv_results_)
        print(results)
        #plt.figure()
        #plt.plot(results.param_n_estimators, results.mean_test_score, 'r.')

        # prediction on test set using the best model
        reg_rf = grid_search.best_estimator_
        y_comp_rf = reg_rf.predict(X_test)
        n_nodes = [t.tree_.node_count for t in reg_rf.estimators_]
        print("Number of trees    {:d}".format(len(n_nodes)))
        print("Number of nodes (mean, min, max)    {:.2f}, {:d}, {:d}".format(np.mean(n_nodes), min(n_nodes), max(n_nodes)))
        print("Training/test set R^2:    {:.4f}/{:.4f}".format(reg_rf.score(X_trainval, y_trainval), reg_rf.score(X_test, y_test)))

    # training RF without optimization
    else:
        # Random forest regression
        start_time = time.time()
        reg_rf = RandomForestRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=150, max_features='sqrt', n_jobs=NJOBS, random_state=42, bootstrap=True, criterion='mse')
        reg_rf.fit(X_trainval, y_trainval)
        print("N jobs: {:d}".format(NJOBS))
        print("Time of building the random forest (sec): {:.3f}".format(time.time() - start_time))
        y_comp_rf = reg_rf.predict(X_test)
        n_nodes = [t.tree_.node_count for t in reg_rf.estimators_]
        print("Number of trees    {:d}".format(len(n_nodes)))
        print("Number of nodes (mean, min, max)    {:.2f}, {:d}, {:d}".format(np.mean(n_nodes), min(n_nodes), max(n_nodes)))
        print("Training/test set R^2:    {:.4f}/{:.4f}".format(reg_rf.score(X_trainval, y_trainval), reg_rf.score(X_test, y_test)))

        filename_rf = 'energy_rf_model.joblib'
        joblib.dump(reg_rf, filename_rf)

    # ENERGY BIAS AND ENERGY RESOLUTION
    e_true = 10**-3 * y_test * param_test['intensity']       # nove MC
    e_comp = 10**-3 * y_comp_rf * param_test['intensity']    #

    e_edges = np.logspace(np.log10(0.01), np.log10(100), 21)    # fixni binning z prod4 - 5 binu na dekadu
    e = (e_edges[:-1] + e_edges[1:])/2.0
    e_log = np.log10(e)
    relative_error = (e_comp - e_true) / e_true

    gauss_fit = []
    err_low = []
    err_up = []
    e_bias_68 = []
    e_res_68 = []
    err_low_68 = []
    err_up_68 = []

    for i in range(len(e_edges)-1):
        mask = (e_true > e_edges[i]) & ((e_true <= e_edges[i+1]))
        if sum(mask) > 10:

            # energy resolution as a band between differences (median-h1, median-h2)
            median = np.median(relative_error[mask])
            h1=np.quantile(relative_error[mask], 0.15865)   # 50 – (68.27 / 2) = 50 – 34.135 = 15.865
            h2=np.quantile(relative_error[mask], 0.84135)   # 50 + (68.27 / 2) = 50 + 34.135 = 84.135
            median = np.quantile(relative_error[mask], 0.5) 
            e_bias_68.append([e[i], median])
            e_res_68.append([e[i], (h2-h1)/2])
            err_low_68.append(e[i] - e_edges[i])
            err_up_68.append(e_edges[i+1] - e[i])

            # gaussian fit
            values, edges = np.histogram(relative_error[mask], bins='auto', range=(-1, 4))
            if len(values) > 3:
                rms = np.sqrt(sum(relative_error[mask]**2)/len(relative_error[mask])) # rms distribuce kolem nuly - tohle je poctivejsi energy resolution, protoze zahrnuje i vliv biasu
                guess = [max(values), 0, 1, 0]
                centers = (edges[:-1] + edges[1:])/2.0
                best_vals, covar = optimize.curve_fit(gaussian, centers, values, p0=guess)
                gauss_fit.append([e[i], best_vals[1], best_vals[2], np.sqrt(covar[1,1]), np.sqrt(covar[2,2]), centers[values == max(values)][0], np.mean(relative_error[mask]), rms])
                err_low.append(e[i] - e_edges[i])
                err_up.append(e_edges[i+1] - e[i])

    e_res_68 = np.array(e_res_68)
    gauss_fit = np.array(gauss_fit)

    # FIGURES

    # Plotting feature importance
    plt.figure()
    feat_importances = pd.Series(reg_rf.feature_importances_, index=X.columns)
    feat_importances.plot(kind='barh')
    plt.tight_layout()
    plt.savefig('energy_features.png', dpi=100)

    # True vs reconstructed energy
    plt.figure(figsize=(6, 5))
    plt.hist2d(np.log10(e_true), np.log10(e_comp), bins=100, range=np.array([(-2.0, 1.0), (-2.0,1.0)]), norm=mpl.colors.LogNorm())
    plt.plot([-2.5, 1.0], [-2.5, 1.0], 'w-', linewidth=1)
    cbar = plt.colorbar()
    cbar.set_label('N of events')
    plt.xlabel('$\mathsf{log_{10}(Energy_{true}) \; [TeV]}$')
    plt.ylabel('$\mathsf{log_{10}(Energy_{comp}) \; [TeV]}$')
    plt.tight_layout()
    plt.savefig('energy_true_rec_hist.png', dpi=100)

    # True energy vs relative error
    plt.figure(figsize=(6, 5))
    plt.hist2d(np.log10(e_true), relative_error, bins=100, range=np.array([(-2.0, 1.0), (-1.5, 2.5)]), norm=mpl.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('N of events')
    plt.xlabel('Log10 Energy_true [TeV]')
    plt.ylabel('(Energy_comp - Energy_true) /Energy_true')
    plt.tight_layout()
    plt.savefig('energy_err_hist.png', dpi=100)

    # Energy bias and resolution in a single plot
    fig, ax1 = plt.subplots(figsize=(6, 5))
    #ax1.errorbar(gauss_fit[:, 0], abs(gauss_fit[:, 2]), yerr=gauss_fit[:, 4], xerr=[err_low, err_up], fmt='b.', label='gauss mu')
    #ax1.plot(gauss_fit[:, 0], abs(gauss_fit[:, 7]), 'b+', label='rms')
    ax1.errorbar(e_res_68[:, 0], e_res_68[:, 1], xerr=[err_low_68, err_up_68], fmt='b.', label='68% containment')
    ax1.set_xlabel('Energy [TeV]')
    ax1.set_xscale('log')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Energy resolution', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([0, 0.5])

    ax2 = ax1.twinx()
    #ax2.errorbar(gauss_fit[:, 0], gauss_fit[:, 1], yerr=gauss_fit[:, 3], xerr=[err_low, err_up], fmt='r.', label='gauss sigma')
    ax2.errorbar(e_bias_68[:, 0], e_bias_68[:, 1], xerr=[err_low, err_up], fmt='m.', label='median')
    ax2.errorbar(gauss_fit[:, 0], gauss_fit[:, 6], xerr=[err_low, err_up], fmt='r.', label='mean')
    #ax2.plot(gauss_fit[:, 0], gauss_fit[:, 5], 'r+', label='max')
    #ax2.plot(gauss_fit[:, 0], gauss_fit[:, 6], 'ro', label='mean')
    ax2.set_ylabel('Energy bias', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim([-0.3, 1.1])
    ax2.set_xlim([0.01, 10])
    plt.tight_layout()

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    plt.savefig('energy_bias.png', dpi=100)

    plt.show()
