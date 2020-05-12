#!/usr/bin/env python

"""
Usage:
  ./all_reco.py --energy_model=rf_model.joblib --disp_model=rf_model.joblib --sep_model=gh_sep_model.joblib --input=dl1b_file.h5 --output=dl2_outfize.h5
"""

import argparse
import pandas as pd
import numpy as np
from lstchain.io import write_dl2_dataframe
from sklearn.externals import joblib
from sklearn.utils import shuffle
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.reco import utils
import tables
from scipy import optimize
import astropy.units as u
from sklearn import metrics


dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'

parser = argparse.ArgumentParser()

parser.add_argument('--energy_model', action='store', type=str,
                    dest='energy_model',
                    default=None
                    )

parser.add_argument('--sep_model', action='store', type=str,
                    dest='sep_model',
                    default=None
                    )

parser.add_argument('--disp_model', action='store', type=str,
                    dest='disp_model',
                    default=None
                    )

parser.add_argument('--input', action='store', type=str,
                    dest='input',
                    default=None
                    )

parser.add_argument('--output', action='store', type=str,
                    dest='output',
                    default=None
                    )

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--telescope', '-t', action='store', type=int,
                    dest='telescope',
                    default=1
                    )

parser.add_argument('--cam_key', action='store', type=str,
                    dest='dl1_params_camera_key',
                    default=dl1_params_lstcam_key
                    )

args = parser.parse_args()

def quality_cuts(data,
                 intensity_cut=100,
                 leak_cut=0.1,
                 islands_cut=2,
                 ):
    #mask = (10**data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    data_masked = data[mask]
    return data_masked

def data_prepare(filename, key=dl1_params_lstcam_key, filters=None, telescope=1, quality=True):

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
        param = quality_cuts(param)
        param = param.dropna()
        print('Size of dataset after selection cuts:', param.shape[0])
    else: print('No quality cuts applied')

    param = shuffle(param).reset_index(drop=True)
    return param

def apply_models(dl1, reg_energy, reg_disp, class_sep, energy_features, disp_features, class_features):

    X_energy = dl1[energy_features]
    y_comp_rf_energy = reg_energy.predict(X_energy)
    reco_energy = 10**-3 * y_comp_rf_energy * dl1['intensity']  # TeV

    X_disp = dl1[disp_features]
    reco_dXY_disp = reg_disp.predict(X_disp)
    x_reco = dl1['x'] + reco_dXY_disp[:, 0]
    y_reco = dl1['y'] + reco_dXY_disp[:, 1]

    X_gh = dl1[class_features]
    gammaness = class_sep.predict_proba(X_gh)

    dl2 = dl1.copy()
    dl2['reco_energy'] = reco_energy
    dl2['reco_src_x'] = x_reco
    dl2['reco_src_y'] = y_reco
    dl2['reco_disp_dx'] = reco_dXY_disp[:, 0]
    dl2['reco_disp_dy'] = reco_dXY_disp[:, 1]
    dl2['gammaness'] = gammaness[:, 1]

    return dl2


if __name__ == '__main__':

    filename = args.input
    energy_model = args.energy_model
    disp_model = args.disp_model
    sep_model = args.sep_model
    outfile = args.output

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)
    events_filters = config["events_filters"]

    # nacteni pouze sloupcu s parametry do pandas dataframe
    param = data_prepare(filename, key=args.dl1_params_camera_key, filters=events_filters, telescope=args.telescope, quality=True)

    # Reconstruction
    energy_rf_model = joblib.load(energy_model)
    disp_rf_model = joblib.load(disp_model)
    gh_sep_rf_model = joblib.load(sep_model)

    # timing
    energy_features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis']
    disp_features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']
    sep_features = ['log_intensity', 'leakage', 'length', 'width', 'time_gradient', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']

    # no timing
    #energy_features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis']
    #disp_features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']
    #sep_features = ['log_intensity', 'leakage', 'length', 'width', 'skewness', 'kurtosis', 'phi', 'psi', 'wl']


    dl2 = apply_models(param, energy_rf_model, disp_rf_model, gh_sep_rf_model, energy_features, disp_features, sep_features)

    # Save dl2 data into a file
    dl2.to_hdf(outfile, key=args.dl1_params_camera_key)
