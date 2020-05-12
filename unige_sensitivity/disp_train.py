#!/usr/bin/env python

"""
Usage:
  ./disp_train.py --optimize --fast --train_file=train_file.h5 --test_file=test_file.h5
  ./disp_train.py --train_file=train_file.h5 --test_file=test_file.h5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.reco import utils
import astropy.units as u
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from scipy import optimize
#import mglearn

dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'


parser = argparse.ArgumentParser()

parser.add_argument('--optimize', '-o', dest='optimize', action='store_const', default=False, const=True)

parser.add_argument('--fast', '-f', dest='fast', action='store_const', default=False, const=True)

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

if __name__ == '__main__':

    set_figures()

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

    # Aplikace dodatecneho cutu na vzdalenost zdroje pro training dataset
    src_center_distance = np.sqrt(param_train['src_x']**2 + param_train['src_y']**2)
    cut_deg = 1 # deg
    mask = src_center_distance < cut_deg/2.0
    param_train = param_train[mask]
    print('Size of training dataset after source position cut:', param_train.shape[0])

    # source position in difuse training data
    x_src = param_train['src_x']
    y_src = param_train['src_y']

    # transformation to prime coordinates - dx' is along true main axis of hillas elipse (defined by disp_angle), dy' is ortogonal to this axis, so the true value is 0
    param_train['disp_dx_prime'] = param_train['disp_dx'] / np.cos(param_train['disp_angle'])
    param_test['disp_dx_prime'] = param_test['disp_dx'] / np.cos(param_test['disp_angle'])
    param_train['disp_dy_prime'] = 0 * param_train['disp_dy']
    param_test['disp_dy_prime'] = 0 * param_test['disp_dy']

    features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']
    #features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']     # no timing


    # Split into features and target
    X = param_train[features]
    target = np.transpose([param_train[f] for f in ['disp_dx', 'disp_dy']])

    # split data into TRAINING / VALIDATION
    X_trainval = X
    y_trainval = target

    X_test = param_test[features]
    y_test = np.transpose([param_test[f] for f in ['disp_dx', 'disp_dy']])

    print('X_train+validation shape: ', X_trainval.shape[0])
    print('X_test shape: ', X_test.shape[0])

    # Number of cores used
    NJOBS=16

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
        filename_rf = 'disp_rf_model.joblib'
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

        filename_rf = 'disp_rf_model.joblib'
        joblib.dump(reg_rf, filename_rf)

    # training RF without optimization
    else:
        # Random forest regression
        reg_rf = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=150, max_features='sqrt', n_jobs=NJOBS, random_state=42, bootstrap=True, criterion='mse')
        reg_rf.fit(X_trainval, y_trainval)
        y_comp_rf = reg_rf.predict(X_test)
        print("RF: Training/test set R^2:    {:.4f}/{:.4f}".format(reg_rf.score(X_trainval, y_trainval), reg_rf.score(X_test, y_test)))

        filename_rf = 'disp_rf_model.joblib'
        joblib.dump(reg_rf, filename_rf)

    # reco source position on test data
    # - transformation to np.array is neccessary
    dx_reco = y_comp_rf[:, 0]
    dy_reco = y_comp_rf[:, 1]
    x_reco = np.array(param_test['x'] + dx_reco) * u.m
    y_reco = np.array(param_test['y'] + dy_reco) * u.m
    focal_length = 28 * u.m
    pointing_alt = np.array(param_test['mc_alt_tel']) * u.rad
    pointing_az = np.array(param_test['mc_az_tel']) * u.rad

    if args.fast:
        # TEMPORARY - just to speed up the code for tests as astropy transformations takes ages
        #################################################
        f = 0.1 / 0.05  # deg / m
        src_separation_approx = f * np.sqrt((x_reco.value - np.array(param_test['src_x'])[0])**2 + (y_reco.value - np.array(param_test['src_y'])[0])**2)
        sorted = np.sort(src_separation_approx)
        index_68 = int(0.6827 * sorted.size)
        resolution_approx = sorted[index_68] * u.deg
        print("APPROXIMATIVE Resolution (68% containment): {:.4f}".format(resolution_approx))
        ################################################
    else:
        # Resolution for all energy bins together
        # transformation of reconstructed coordinates to horizon frame: (az, alt) in deg
        # - transformace vcetne vypoctu .separation a rozliseni funguje dobre - overeno aproximacnim
        #   vypoctem pomoci prepoctu vzdalenosti v metrech s pomoci faktoru 0.1 deg / 0.05 m, coz
        #   je rozliseni a velikost jednoho pixelu LST kamery
        sky_coords_reco = utils.camera_to_sky(x_reco, y_reco, focal_length,  pointing_alt, pointing_az)
        # Transformation of true source coordinates to horizon frame
        sky_coords_source = utils.camera_to_sky(np.array(param_test['src_x'])[0] * u.m, np.array(param_test['src_y'])[0] * u.m, focal_length,  pointing_alt[0], pointing_az[0])
        # Angular distance of each reconstructed position to the true position of the source
        src_separation = sky_coords_reco.separation(sky_coords_source)
        # 68 % containment
        sorted = np.sort(src_separation)
        index_68 = int(0.6827 * sorted.size)
        resolution = sorted[index_68]
        print("Resolution (68% containment): {:.4f}".format(resolution))


    # FIGURES

    # Plotting feature importance
    plt.figure()
    feat_importances = pd.Series(reg_rf.feature_importances_, index=X.columns)
    feat_importances.plot(kind='barh')
    plt.tight_layout()
    plt.savefig('disp_features.png', dpi=100)

    # True vs reconstructed disp
    plt.figure(figsize=(6, 5))
    plt.hist2d(y_test[:, 0], y_comp_rf[:, 0], bins=100, range=np.array([(-1.5, 1.5), (-1.5, 1.5)]), norm=mpl.colors.LogNorm())
    plt.plot([-1.5, 1.5], [-1.5, 1.5], 'w-', linewidth=1)
    cbar = plt.colorbar()
    cbar.set_label('N of events')
    plt.xlabel('DISP dX true')
    plt.ylabel('DISP dX comp')
    plt.tight_layout()
    plt.savefig('disp_dx_true_rec_hist.png', dpi=100)

    plt.figure(figsize=(6, 5))
    plt.hist2d(y_test[:, 1], y_comp_rf[:, 1], bins=100, range=np.array([(-1.5, 1.5), (-1.5, 1.5)]), norm=mpl.colors.LogNorm())
    plt.plot([-1.5, 1.5], [-1.5, 1.5], 'w-', linewidth=1)
    cbar = plt.colorbar()
    cbar.set_label('N of events')
    plt.xlabel('DISP dY true')
    plt.ylabel('DISP dY comp')
    plt.tight_layout()
    plt.savefig('disp_dy_true_rec_hist.png', dpi=100)


    if args.fast:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.hist2d(x_reco, y_reco, bins=100, range=np.array([(-1.5, 1.5), (-1.5, 1.5)]), norm=mpl.colors.LogNorm())
        plt.plot(param_test['src_x'][0], param_test['src_y'][0], 'r+')
        circle = mpl.patches.Circle((param_test['src_x'][0], param_test['src_y'][0]), radius=resolution_approx.value / f, edgecolor='red', linewidth=2, facecolor='None')
        ax.add_patch(circle)
        cbar = plt.colorbar()
        cbar.set_label('N of events')
        plt.title('Camera frame')
        plt.xlabel('x source reco (testing) [m]')
        plt.ylabel('y source reco (testing) [m]')
        plt.tight_layout()
        plt.savefig('disp_src_reco_hist.png', dpi=100)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        factor = np.cos(sky_coords_source.alt.rad)
        plt.hist2d(sky_coords_reco.az * factor, sky_coords_reco.alt, bins=100, norm=mpl.colors.LogNorm())
        plt.plot(sky_coords_source.az * factor, sky_coords_source.alt, 'r+')
        circle = mpl.patches.Circle((sky_coords_source.az.value * factor, sky_coords_source.alt.value), radius=resolution.value, edgecolor='red', linewidth=2, facecolor='None')
        ax.add_patch(circle)
        cbar = plt.colorbar()
        cbar.set_label('N of events')
        plt.title('Horizon frame')
        plt.xlabel('az source reco * cos(alt) (testing) [deg]')
        plt.ylabel('alt source reco (testing) [deg]')
        plt.tight_layout()
        plt.savefig('disp_src_reco_deg_hist.png', dpi=100)

