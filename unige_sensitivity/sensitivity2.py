#!/usr/bin/env python

"""
    sensitivity2.py --input_gamma=dl2_gamma.h5 --input_proton=dl2_proton.h5 --weights_gamma=gamma_weights.hd5 --weights_proton=proton_weights.hd5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import astropy.units as u
import scipy.stats as stats
from scipy.optimize import least_squares
from ctapipe.io import HDF5TableReader
from lstchain.io.lstcontainers import ThrownEventsHistogram
from lstchain.io.io import read_simtel_energy_histogram, read_simu_info_hdf5
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.reco import utils
from tqdm import trange

dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'
parser = argparse.ArgumentParser()

parser.add_argument('--input_gamma', action='store', type=str,
                    dest='input_gamma',
                    default=None
                    )

parser.add_argument('--input_gamma_diffuse', action='store', type=str,
                    dest='input_gamma_diffuse',
                    default=None
                    )
parser.add_argument('--input_proton', action='store', type=str,
                    dest='input_proton',
                    default=None
                    )

parser.add_argument('--weights_gamma', action='store', type=str,
                    dest='weights_gamma',
                    default=None
                    )

parser.add_argument('--weights_proton', action='store', type=str,
                    dest='weights_proton',
                    default=None
                    )

parser.add_argument('--cam_key', '-k', action='store', type=str,
                    dest='dl1_params_camera_key',
                    default=dl1_params_lstcam_key
                    )

parser.add_argument('--intensity_min', '-i', action='store', type=float,
                    dest='intensity_cut',
                    default=300
                    )
parser.add_argument('--fast', '-f', dest='fast', action='store_const', default=False, const=True)

args = parser.parse_args()

def set_figures():

    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['legend.numpoints'] = 1 #aby se u errorbaru a bodovych grafu nezobrazoval kazdy bod 2x
    mpl.rcParams['lines.markersize'] = 15
    mpl.rcParams['legend.fontsize'] = 12

def sigma_lima(N_on, N_off, alpha):
    sigma_lima = np.sqrt(2 * (N_on * np.log((1+alpha)/alpha * N_on / (N_on + N_off)) + N_off * np.log((1+alpha) * (N_off / (N_on + N_off))) ))
    return sigma_lima

# Aleksic et al., 2015, JHEAP, (https://doi.org/10.1016/j.jheap.2015.01.002), fit od 0.1-20 TeV
def crab_magic2015_dNdE(energy):
    spectrum = 3.23 * 10**-11 * energy**(-2.47 - 0.24 * np.log10(energy))   # TeV-1 cm-2 s-1
    #spectrum = spectrum / 1.602176487   # erg-1 cm-2 s-1
    spectrum = spectrum * 10000 # TeV-1 m-2 s-1
    return spectrum

def crab_hegra_dNdE(energy):
    const = 2.83*10**-7 # TeV^-1 m^-2 s^-1
    spectral_index_real = -2.62
    e0 = 1 # TeV
    spectrum = const * (energy / e0)**spectral_index_real
    return spectrum

# tohle se musi nasobit hegra spektrem, protoze v tom je to pocitany
def sensitivity_comp(N_gamma, N_proton, alpha, mid_energy):

    # pozadavek na 5 sigma detekci
    sigma = sigma_lima(N_gamma + alpha*N_proton, N_proton, alpha)
    #sigma = (N_gamma)/np.sqrt(N_proton * alpha)
    sensitivity_5sig = 5/sigma * crab_hegra_dNdE(mid_energy)  # frakce toku craba, ktery muzeme videt na 5 sigma v danem energetickem binu

    # pozadavek na N_gamma > 10
    sensitivity_10gamma = 10/N_gamma * crab_hegra_dNdE(mid_energy)

    # pozadavek na N_gamma > 0.05 * N_bkg
    sensitivity_005bkg = 0.05 * (N_proton*alpha) / N_gamma * crab_hegra_dNdE(mid_energy)

    # celkova sensitivita
    sensitivity = []
    for i in range(len(sensitivity_5sig)):
        sensitivity.append(
            max(
                sensitivity_5sig[i], 
                sensitivity_10gamma[i], 
                sensitivity_005bkg[i]
            )
        )
    sensitivity = np.array(sensitivity)
    sensitivity[sensitivity == 0] = np.nan

    return sensitivity, sensitivity_5sig, sensitivity_10gamma, sensitivity_005bkg


def compute_energy_resolution(relative_error_energy, e_true, energy_bins, plot='resolution_energy.png', txtfile='resolution_energy.txt'):
    mid_energy = 0.5 * (energy_bins[1:] + energy_bins[:-1])
    de_low = mid_energy - energy_bins[:-1]
    de_up = energy_bins[1:] - mid_energy
    print('mid_energy:', mid_energy)
    print('de_low:', de_low)
    print('de_up:', de_up)
    e_resol68 = np.zeros_like(mid_energy)
    e_bias = np.zeros_like(mid_energy)
    for i in range(len(energy_bins)-1):
        mask = (e_true > energy_bins[i]) & ((e_true <= energy_bins[i+1]))
        if np.sum(mask) > 10:
            e_bias[i] = np.quantile(relative_error_energy[mask], 0.5) 
            h1=np.quantile(relative_error_energy[mask], 0.15865)   # 50 – (68.27 / 2) = 50 – 34.135 = 15.865
            h2=np.quantile(relative_error_energy[mask], 0.84135)   # 50 + (68.27 / 2) = 50 + 34.135 = 84.135
            e_resol68[i] = (h2-h1)/2
        else:
            e_resol68[i] = np.nan
            e_bias[i] = np.nan
    
    if plot is not None:
        fig, ax1 = plt.subplots(figsize=(6, 5))
        ax1.errorbar(mid_energy, e_resol68, xerr=[de_low, de_up], fmt='b.', label='68% containment')
        ax1.set_xlabel('Energy [TeV]')
        ax1.set_xscale('log')
        ax1.set_ylabel('Energy resolution', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim([0.03, 10])
        ax1.set_ylim([0, 0.5])

        ax2 = ax1.twinx()
        ax2.errorbar(mid_energy, e_bias, xerr=[de_low, de_up], fmt='r.', label='median')
        ax2.set_ylabel('Energy bias', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_ylim([-0.3, 1.1])
        ax2.set_xlim([0.03, 10])
        plt.tight_layout()
        plt.savefig(plot)
    if txtfile is not None:
        data = np.stack([energy_bins[:-1], energy_bins[1:], e_resol68, e_bias])
        np.savetxt(txtfile, data.T, header='Emin_bin Emax_bin E_res_68 E_bias')


def compute_resolution_point_source(error_distance_deg, energy, energy_bins, plot='resolution_68.png', txtfile='resolution_68.txt'):
    theta_bins = np.linspace(0, 0.5, 51)
    mid_energy = 0.5 * (energy_bins[1:] + energy_bins[:-1])
    mid_theta = 0.5 * (theta_bins[1:] + theta_bins[:-1])
    count_histo, _, _ = np.histogram2d(energy, error_distance_deg, bins=[energy_bins, theta_bins])
    count_per_energy = np.sum(count_histo, axis=1, keepdims=True)
    fraction_worse_theta_res = np.zeros_like(count_histo) * np.NaN
    filled_energy_bin = count_per_energy.flatten() > 10
    fraction_worse_theta_res[filled_energy_bin, :] = np.flip(
        np.cumsum(
            np.flip(count_histo[filled_energy_bin, :], axis=1), axis=1
        ) / count_per_energy[filled_energy_bin, :], 
        axis=1
    )
    resolution_68 = np.zeros_like(mid_energy)
    for b in range(len(mid_energy)):
        if not filled_energy_bin[b]:
            resolution_68[b] = 1
            continue
        resolution_68[b] = np.interp(
            1-0.682, np.flip(fraction_worse_theta_res[b, :]), np.flip(mid_theta)
        )
    if plot is not None:
        fig = plt.figure()
        X, Y = np.meshgrid(energy_bins, theta_bins)
        plt.pcolormesh(X, Y, fraction_worse_theta_res.T * 100)
        plt.step(mid_energy, resolution_68, 'r', label=r'$\Theta_{68}$', where='mid')
        plt.xscale('log')
        plt.xlabel(r'$E_{MC}$ (TeV)')
        plt.ylabel(r'$\Theta_{res}$ (deg)')
        plt.ylim([theta_bins[0], theta_bins[-1]])
        cbar = plt.colorbar()
        cbar.set_label(r'% event with $|\Theta_{rec} - \Theta_{true}| > \Theta_{res}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot)
        plt.close(fig)
    if txtfile is not None:
        data = np.stack([energy_bins[:-1], energy_bins[1:], resolution_68])
        np.savetxt(txtfile, data.T, header='Emin_bin Emax_bin resolution( 68% containment)')
    return resolution_68


def compute_resolution_off_axis(error_distance_deg, energy, offaxis_angle, energy_bins, offaxis_angle_bins, plot='resolution_68_offaxis.png', txtfile='resolution_68_offaxis.txt'):
    energy_bins = np.array(energy_bins)
    offaxis_angle_bins = np.array(offaxis_angle_bins)
    n_offaxis_angle_bin = offaxis_angle_bins.shape[0] - 1
    n_energy_bin = energy_bins.shape[0] - 1
    resol_offaxis = np.zeros([n_offaxis_angle_bin, n_energy_bin])
    labels=[]
    for b in range(n_offaxis_angle_bin):
        in_offaxis_angle_bin = np.logical_and(
            offaxis_angle >= offaxis_angle_bins[b], offaxis_angle < offaxis_angle_bins[b+1]
        )
        resol_offaxis[b, :] = compute_resolution_point_source(
            error_distance_deg[in_offaxis_angle_bin], 
            energy[in_offaxis_angle_bin], 
            energy_bins, plot=None, txtfile=None
        )
        labels.append(f'{offaxis_angle_bins[b]:.2}<=off_axis(deg)<{offaxis_angle_bins[b+1]:.2} ({np.sum(in_offaxis_angle_bin)} events)')
    if plot is not None:
        fig = plt.figure()
        for b in range(n_offaxis_angle_bin):
            plt.step(mid_energy, resol_offaxis[b, :], where='mid', label=labels[b])
        plt.xscale('log')
        plt.xlabel(r'$E_{MC}$ (TeV)')
        plt.ylabel(r'$\Theta_{68}$ (deg)')
        plt.legend()
        plt.ylim([0, 1])
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot)
        plt.close(fig)
    if txtfile is not None:
        data = [energy_bins[:-1], energy_bins[1:]]
        header='Emin_bin Emax_bin '
        for b in range(n_offaxis_angle_bin):
            header += ' {labels[b]}'
            data.append(resol_offaxis[b, :])
        data=np.stack(data)
        np.savetxt(txtfile, data.T, header=header)


def plot_magic_sensitivity():
    magic_sens = np.array([0.068, 0.0367, 0.0225, 0.0203, 0.0151, 0.0153, 0.0166, 0.0236, 0.038, 0.061, 0.15])    # in % of Crab
    log10_energy_tev = [-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    energy = 10**np.array(log10_energy_tev)
    plt.plot(energy, energy**2 * magic_sens * crab_magic2015_dNdE(energy), 'k-', label='MAGIC stereo')


def quality_cuts(data,
                 intensity_cut=300,
                 leak_cut=0.1,
                 islands_cut=2,
                 ):
    print('quality cuts: intensity >=', intensity_cut, ', leak <=', leak_cut, ', island <=', islands_cut)
    mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    data_masked = data[mask]
    return data_masked


if __name__ == '__main__':

    set_figures()

    filename_g = args.input_gamma
    filename_gd = args.input_gamma_diffuse
    filename_p = args.input_proton

    #read weights and dl2 info for protons
    energy_bins_proton = pd.read_hdf(args.weights_proton, mode='r', key='energy_bins').values  # TeV
    weights_proton = pd.read_hdf(args.weights_proton, mode='r', key='weights').values
    thrown_hist_proton = pd.read_hdf(args.weights_proton, mode='r', key='thrown_hist').values
    param_proton = pd.read_hdf(filename_p, key=args.dl1_params_camera_key)
    #read weights and dl2 info for point source gamma
    energy_bins_gamma = pd.read_hdf(args.weights_gamma, mode='r', key='energy_bins').values    # TeV
    weights_gamma = pd.read_hdf(args.weights_gamma, mode='r', key='weights').values
    thrown_hist_gamma = pd.read_hdf(args.weights_gamma, mode='r', key='thrown_hist').values
    param_gamma = pd.read_hdf(filename_g, key=args.dl1_params_camera_key)
    #compute weigths to get rates out of histograms
    param_gamma['weights'] = np.ones(param_gamma['width'].size)
    param_proton['weights'] = np.ones(param_proton['width'].size)
    for i in range(len(energy_bins_gamma)-1):
        mask = (param_gamma['mc_energy'] > energy_bins_gamma[i]) & (param_gamma['mc_energy'] <= energy_bins_gamma[i+1])
        param_gamma['weights'][mask] = weights_gamma[i]
        mask = (param_proton['mc_energy'] > energy_bins_proton[i]) & (param_proton['mc_energy'] <= energy_bins_proton[i+1])
        param_proton['weights'][mask] = weights_proton[i]
    #get point source gamma and proton rate function of energy before quality cuts
    param_gamma_energy_hist_dl2 = np.histogram(param_gamma['reco_energy'], bins=energy_bins_gamma, weights=param_gamma['weights'])[0]
    param_proton_energy_hist_dl2 = np.histogram(param_proton['reco_energy'], bins=energy_bins_proton, weights=param_proton['weights'])[0]
    
    #apply quality cuts
    print(len(param_gamma), 'gamma events before quality cuts')
    param_gamma= quality_cuts(param_gamma, intensity_cut=args.intensity_cut)
    print(len(param_gamma), 'gamma events after quality cuts')
    param_gamma_diffuse = pd.read_hdf(filename_gd, key=args.dl1_params_camera_key)
    print(len(param_gamma_diffuse), 'diffuse gamma events before quality cuts')
    param_gamma_diffuse= quality_cuts(param_gamma_diffuse, intensity_cut=args.intensity_cut)
    print(len(param_gamma_diffuse), 'diffuse gamma events after quality cuts')
    print(len(param_proton), 'proton events before quality cuts')
    param_proton= quality_cuts(param_proton, intensity_cut=args.intensity_cut)
    print(len(param_proton), 'proton events after quality cuts')
    #get point source gamma and proton rate function of energy after quality cuts
    param_gamma_energy_hist_qc = np.histogram(param_gamma['reco_energy'], bins=energy_bins_gamma, weights=param_gamma['weights'])[0]
    param_proton_energy_hist_qc = np.histogram(param_proton['reco_energy'], bins=energy_bins_proton, weights=param_proton['weights'])[0]

    # Doplneni dat o parametr theta2
    source_position = [np.array(param_gamma['src_x'])[0], np.array(param_gamma['src_y'])[0]] # source position in the FOV [m]
    pointing_alt_gamma = np.array(param_gamma['mc_alt_tel']) * u.rad
    pointing_az_gamma = np.array(param_gamma['mc_az_tel']) * u.rad
    pointing_alt_gamma_diffuse = np.array(param_gamma_diffuse['mc_alt_tel']) * u.rad
    pointing_az_gamma_diffuse = np.array(param_gamma_diffuse['mc_az_tel']) * u.rad
    pointing_alt_proton = np.array(param_proton['mc_alt_tel']) * u.rad
    pointing_az_proton = np.array(param_proton['mc_az_tel']) * u.rad
    focal_length = 28 * u.m
    if args.fast:
        # TEMPORARY - rychly vypocet bez transformace souradnic
        f = 0.1 / 0.05  # deg / m
        param_gamma['theta2'] = f**2 * ((source_position[0]-param_gamma['reco_src_x'])**2 + (source_position[1]-param_gamma['reco_src_y'])**2)
        param_proton['theta2'] = f**2 * ((source_position[0]-param_proton['reco_src_x'])**2 + (source_position[1]-param_proton['reco_src_y'])**2)
        param_proton['theta2_true'] = f**2 * ((source_position[0]-param_proton['src_x'])**2 + (source_position[1]-param_proton['src_y'])**2)
    else:
        sky_coords_reco_gamma = utils.camera_to_sky(np.array(param_gamma['reco_src_x']) * u.m, np.array(param_gamma['reco_src_y']) * u.m, focal_length,  pointing_alt_gamma, pointing_az_gamma)
        sky_coords_true_gamma = utils.camera_to_sky(np.array(param_gamma['src_x']) * u.m, np.array(param_gamma['src_y']) * u.m, focal_length,  pointing_alt_gamma, pointing_az_gamma)
        sky_coords_reco_gamma_diffuse = utils.camera_to_sky(np.array(param_gamma_diffuse['reco_src_x']) * u.m, np.array(param_gamma_diffuse['reco_src_y']) * u.m, focal_length,  pointing_alt_gamma_diffuse, pointing_az_gamma_diffuse)
        sky_coords_true_gamma_diffuse = utils.camera_to_sky(np.array(param_gamma_diffuse['src_x']) * u.m, np.array(param_gamma_diffuse['src_y']) * u.m, focal_length,  pointing_alt_gamma_diffuse, pointing_az_gamma_diffuse)
        sky_coords_reco_proton = utils.camera_to_sky(np.array(param_proton['reco_src_x']) * u.m, np.array(param_proton['reco_src_y']) * u.m, focal_length,  pointing_alt_proton, pointing_az_proton)
        sky_coords_true_proton = utils.camera_to_sky(np.array(param_proton['src_x']) * u.m, np.array(param_proton['src_y']) * u.m, focal_length,  pointing_alt_proton, pointing_az_proton)
        sky_coords_source_gamma = utils.camera_to_sky(source_position[0] * u.m, source_position[1] * u.m, focal_length,  pointing_alt_gamma[0], pointing_az_gamma[0])
        sky_coords_source_gamma_diffuse = utils.camera_to_sky(source_position[0] * u.m, source_position[1] * u.m, focal_length,  pointing_alt_gamma[0], pointing_az_gamma[0])
        sky_coords_source_proton = utils.camera_to_sky(source_position[0] * u.m, source_position[1] * u.m, focal_length,  pointing_alt_proton[0], pointing_az_proton[0])
        sky_coords_camera_center =  utils.camera_to_sky(0 * u.m, 0 * u.m, focal_length,  pointing_alt_gamma_diffuse[0], pointing_az_gamma_diffuse[0])
        param_gamma['offaxis_angle'] = sky_coords_true_gamma.separation(sky_coords_camera_center)
        param_gamma_diffuse['offaxis_angle'] = sky_coords_true_gamma_diffuse.separation(sky_coords_camera_center)
        param_proton['offaxis_angle'] = sky_coords_true_proton.separation(sky_coords_camera_center)
        # Theta2
        param_gamma['theta2'] = sky_coords_reco_gamma.separation(sky_coords_source_gamma)**2
        param_gamma['theta2_true'] = sky_coords_true_gamma.separation(sky_coords_source_gamma)**2
        param_gamma['error'] = sky_coords_reco_gamma.separation(sky_coords_true_gamma)
        param_gamma_diffuse['theta2'] = sky_coords_reco_gamma_diffuse.separation(sky_coords_source_gamma_diffuse)**2
        param_gamma_diffuse['theta2_true'] = sky_coords_true_gamma_diffuse.separation(sky_coords_source_gamma_diffuse)**2
        param_gamma_diffuse['error'] = sky_coords_reco_gamma_diffuse.separation(sky_coords_true_gamma_diffuse)
        param_proton['theta2'] = sky_coords_reco_proton.separation(sky_coords_source_proton)**2
        param_proton['theta2_true'] = sky_coords_true_proton.separation(sky_coords_source_proton)**2

    # Bining pro vypocet sensitivity
    # - nemusi mit nic spolecneho s jemnym biningem ve kterem mam napocitane vahy
    # protoze ty mam ted zvlast pro kazdy event
    energy_bins = np.logspace(np.log10(0.01), np.log10(100), 21)   # 5 binu na dekadu (4 rady * 5 + 1)
    mid_energy = (energy_bins[:-1] + energy_bins[1:])/2.0
    err_low = mid_energy - energy_bins[:-1]
    err_up = energy_bins[1:] - mid_energy

    theta_bins = np.linspace(0, 2, 41)
    mid_theta = (theta_bins[:-1] + theta_bins[1:])/2.0

    # TEST jak vypadaji theta distribuce protonu kolem zname pozice zdroje
    param_proton_thetahist = np.histogram(param_proton['theta2'], bins=theta_bins, weights=param_proton['weights'])[0]
    param_proton_thetahist_true = np.histogram(param_proton['theta2_true'], bins=theta_bins, weights=param_proton['weights'])[0]

    # event rate as a function of theta2
    fig=plt.figure()
    plt.title('E [TeV]: all')
    plt.step(mid_theta, param_proton_thetahist, where='mid', label='protons reco')
    plt.step(mid_theta, param_proton_thetahist_true, where='mid', label='protons true')
    plt.ylabel('Event Rate [Hz]')
    plt.xlabel('Theta2 [deg2]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('theta2_protons_all.png')
    plt.close(fig)

    fig = plt.figure()
    plt.title('After g/h separation')
    cuts = np.linspace(0.5, 0.9, 4)
    for cut in cuts:
        reco_mask_proton = param_proton['gammaness'] > cut
        param_proton_thetahist_gh = np.histogram(param_proton[reco_mask_proton]['theta2'], bins=theta_bins, weights=param_proton[reco_mask_proton]['weights'])[0]
        plt.step(mid_theta, param_proton_thetahist_gh, where='mid', label='gh cut '+str(round(cut, 1)))
    plt.ylabel('Event Rate [Hz]')
    plt.xlabel('Theta2 [deg2]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('theta2_protons_all_gh_cuts.png')
    plt.close(fig)

    ###############################
    # optimalizace cutu na nejlepsi sensitivitu pro kazdy bin v energii
    sensitivity_cuts = []
    cuts = []
    N_sim_gammas = []
    N_sim_protons = []
    T = 50 * 60 * 60
    gh_thresholds = np.linspace(0.1, 1, 101)
    # Hodne hrube nastavene cuty na theta2 pro gamma na zaklade krivky pro rozliseni s cuty 300, 0.1, 2, 1deg

    energy_error = param_gamma['reco_energy'] - param_gamma['mc_energy']
    relative_energy_error = energy_error / param_gamma['mc_energy']
    compute_energy_resolution(relative_energy_error, param_gamma['mc_energy'], energy_bins, plot='resolution_energy.png', txtfile='resolution_energy.txt')
    
    resolution_68 = compute_resolution_point_source(param_gamma['error'], param_gamma['mc_energy'] , energy_bins)
    compute_resolution_off_axis(param_gamma_diffuse['error'], param_gamma_diffuse['mc_energy'], param_gamma_diffuse['offaxis_angle'], energy_bins, offaxis_angle_bins=[0, 0.25, 0.5, 1, 2, 3], plot='resolution_68_offaxis.png', txtfile='resolution_68_offaxis.txt')

    theta2_gamma_cuts = resolution_68**2
    # energeticky zavisle cuty na theta2 podle rozliseni
    reco_mask_gamma_theta2 = np.ones(len(param_gamma['gammaness']), dtype=bool)
    reco_mask_proton_theta2 = np.ones(len(param_proton['gammaness']), dtype=bool)
    for i in range(len(mid_energy)):
        mask_energy_gamma = (param_gamma['reco_energy'] > energy_bins[i]) & (param_gamma['reco_energy'] < energy_bins[i+1])
        mask_energy_proton = (param_proton['reco_energy'] > energy_bins[i]) & (param_proton['reco_energy'] < energy_bins[i+1])
        reco_mask_gamma_theta2[mask_energy_gamma] = (param_gamma[mask_energy_gamma]['theta2'] < theta2_gamma_cuts[i])
        reco_mask_proton_theta2[mask_energy_proton] = (param_proton[mask_energy_proton]['theta2'] < theta2_gamma_cuts[i])
    for i in trange(len(gh_thresholds)):
        reco_mask_proton = (param_proton['gammaness'] > gh_thresholds[i]) & reco_mask_proton_theta2
        reco_mask_gamma = (param_gamma['gammaness'] > gh_thresholds[i]) & reco_mask_gamma_theta2

        # pocty simulovanych eventu po cutech
        gamma_N = np.histogram(param_gamma[reco_mask_gamma]['reco_energy'], bins=energy_bins)[0]
        proton_N = np.histogram(param_proton[reco_mask_proton]['reco_energy'], bins=energy_bins)[0]

        # vahovane histogramy v energii s aplikovanou g/h separaci a theta2 cutem
        gamma_hz = np.histogram(param_gamma[reco_mask_gamma]['reco_energy'], bins=energy_bins, weights=param_gamma[reco_mask_gamma]['weights'])[0]
        proton_hz = np.histogram(param_proton[reco_mask_proton]['reco_energy'], bins=energy_bins, weights=param_proton[reco_mask_proton]['weights'])[0]

        N_gamma = gamma_hz * T
        N_proton = 5 * proton_hz * T    # toto je nejakej abelarduv trik, nevim proc to proste neni 1* a alpha=1
        alpha = 0.2
        sens, s, ss, sss = sensitivity_comp(N_gamma, N_proton, alpha, mid_energy)
        sensitivity_cuts.append(sens)
        cuts.append(gh_thresholds[i])
        N_sim_gammas.append(gamma_N)
        N_sim_protons.append(proton_N)
    sensitivity_cuts = np.array(sensitivity_cuts)
    cuts = np.array(cuts)
    N_sim_gammas = np.array(N_sim_gammas)
    N_sim_protons = np.array(N_sim_protons)

    # Vyber optimalni sensitivity pro kazdy bin v energii
    n_energy_bin = sensitivity_cuts.shape[1]
    #inds_nan = np.where(np.isnan(sensitivity_cuts))
    #sensitivity_cuts[inds_nan] = np.nan
    #indices_min = np.nanargmin(sensitivity_cuts, axis=0)
    sensitivity_optimized = []
    optimized_cuts = []
    print('Optimal cuts')
    #for i in range(len(indices_min)):
    for i in range(n_energy_bin):
        if np.all(np.isnan(sensitivity_cuts[:, i])):
            sensitivity_optimized.append(np.nan)
            optimized_cuts.append([energy_bins[i], energy_bins[i+1], cuts[0]])
            print('E [TeV]: {:.3f}-{:.3f}, gammaness > {:.2f}, Ng = {:d}, Np = {:d}'.format(energy_bins[i], energy_bins[i+1],cuts[0], N_sim_gammas[0, i], N_sim_protons[0, i]))
        else:
            index_min = np.nanargmin(sensitivity_cuts[:, i])
            sensitivity_optimized.append(sensitivity_cuts[index_min, i])
            optimized_cuts.append([energy_bins[i], energy_bins[i+1], cuts[index_min]])
            print('E [TeV]: {:.3f}-{:.3f}, gammaness > {:.2f}, Ng = {:d}, Np = {:d}'.format(energy_bins[i], energy_bins[i+1], cuts[index_min], N_sim_gammas[index_min, i], N_sim_protons[index_min, i]))
    sensitivity_optimized = np.array(sensitivity_optimized)
    optimized_cuts = np.array(optimized_cuts)
    #########################################

    # vahovane histogramy v energii
    param_gamma_hist_nogh = np.histogram(param_gamma['reco_energy'], bins=energy_bins, weights=param_gamma['weights'])[0]  # reco_energy i biny jsou v TeV
    param_proton_hist_nogh = np.histogram(param_proton['reco_energy'], bins=energy_bins, weights=param_proton['weights'])[0]
    print('Proton rate BEFORE g/h separation, full FOV, quality cuts [Hz]: {:.2f}'.format(sum(param_proton_hist_nogh)))
    print('Gamma rate BEFORE g/h separation, full FOV, quality cuts [Hz]: {:.2f}'.format(sum(param_gamma_hist_nogh)))

    # Finalni optimalizovane cuty pro kazdy energeticky bin
    mask_gh_gamma = np.ones(len(param_gamma['gammaness']), dtype=bool)
    mask_gh_proton = np.ones(len(param_proton['gammaness']), dtype=bool)
    mask_theta2_gamma = reco_mask_gamma_theta2
    mask_theta2_proton = reco_mask_proton_theta2

    for i in range(len(optimized_cuts)):
        mask_energy = (param_gamma['reco_energy'] > optimized_cuts[i, 0]) & (param_gamma['reco_energy'] < optimized_cuts[i, 1])
        mask_gh_gamma[mask_energy] = param_gamma[mask_energy]['gammaness'] > optimized_cuts[i, 2]
        mask_energy = (param_proton['reco_energy'] > optimized_cuts[i, 0]) & (param_proton['reco_energy'] < optimized_cuts[i, 1])
        mask_gh_proton[mask_energy] = param_proton[mask_energy]['gammaness'] > optimized_cuts[i, 2]

    #get point source gamma and proton rate function of energy after g/h separation
    param_gamma_energy_hist_gh = np.histogram(param_gamma['reco_energy'][mask_gh_gamma], bins=energy_bins_gamma, weights=param_gamma['weights'][mask_gh_gamma])[0]
    param_proton_energy_hist_gh = np.histogram(param_proton['reco_energy'][mask_gh_proton], bins=energy_bins_proton, weights=param_proton['weights'][mask_gh_proton])[0]

    #get point source gamma and proton rate function of energy after theta2 cut
    reco_mask_gamma = mask_gh_gamma & mask_theta2_gamma
    reco_mask_proton = mask_gh_proton & mask_theta2_proton
    param_gamma_energy_hist_reco = np.histogram(param_gamma['reco_energy'][reco_mask_gamma], bins=energy_bins_gamma, weights=param_gamma['weights'][reco_mask_gamma])[0]
    param_proton_energy_hist_reco = np.histogram(param_proton['reco_energy'][reco_mask_proton], bins=energy_bins_proton, weights=param_proton['weights'][reco_mask_proton])[0]
    data_gamma_rate = np.stack([energy_bins_gamma[:-1], energy_bins_gamma[1:], param_gamma_energy_hist_reco])
    np.savetxt(
        'rate_reco_gamma.txt', data_gamma_rate.T, 
        header='Emin_bin[TeV] Emax_bin[TeV] gama_rate[Hz]'
    )
    data_proton_rate = np.stack([energy_bins_gamma[:-1], energy_bins_gamma[1:], param_proton_energy_hist_reco])
    np.savetxt(
        'rate_reco_proton.txt', data_proton_rate.T, 
        header='Emin_bin[TeV] Emax_bin[TeV] proton_rate[Hz]'
    )

    # vahovane histogramy v theta2 s aplikovanou gh separaci
    param_gamma_thetahist_gh = np.histogram(param_gamma[mask_gh_gamma]['theta2'], bins=theta_bins, weights=param_gamma[mask_gh_gamma]['weights'])[0]
    param_proton_thetahist_gh = np.histogram(param_proton[mask_gh_proton]['theta2'], bins=theta_bins, weights=param_proton[mask_gh_proton]['weights'])[0]

    print('Proton rate after g/h separation, full FOV, quality cuts [Hz]: {:.2f}'.format(sum(param_proton_thetahist_gh)))
    print('Gamma rate after g/h separation, full FOV, quality cuts [Hz]: {:.2f}'.format(sum(param_gamma_thetahist_gh)))

    # vahovane histogramy v energii s aplikovanou g/h separaci a cutem na theta2
    param_gamma_hist_gh_theta = np.histogram(param_gamma[reco_mask_gamma]['reco_energy'], bins=energy_bins, weights=param_gamma[reco_mask_gamma]['weights'])[0]
    param_proton_hist_gh_theta = np.histogram(param_proton[reco_mask_proton]['reco_energy'], bins=energy_bins, weights=param_proton[reco_mask_proton]['weights'])[0]

    print('Proton rate after g/h separation, theta2 cut, quality cuts [Hz]: {:.2f}'.format(sum(param_proton_hist_gh_theta)))
    print('Gamma rate after g/h separation, theta2 cut, quality cuts [Hz]: {:.2f}'.format(sum(param_gamma_hist_gh_theta)))


    # intensity
    intensity_bins = np.logspace(np.log10(10), np.log10(1e6), 101)
    gamma_intensityhist_gh = np.histogram(param_gamma[reco_mask_gamma]['intensity'], bins=intensity_bins, weights=param_gamma[reco_mask_gamma]['weights'])[0]
    proton_intensityhist_gh = np.histogram(param_proton[reco_mask_proton]['intensity'], bins=intensity_bins, weights=param_proton[reco_mask_proton]['weights'])[0]

    # length
    length_bins = np.linspace(0, 1, 81)
    gamma_lengthhist_gh = np.histogram(param_gamma[reco_mask_gamma]['length'], bins=length_bins, weights=param_gamma[reco_mask_gamma]['weights'])[0]
    proton_lengthhist_gh = np.histogram(param_proton[reco_mask_proton]['length'], bins=length_bins, weights=param_proton[reco_mask_proton]['weights'])[0]

    # width
    width_bins = np.linspace(0, 0.2, 81)
    gamma_widthhist_gh = np.histogram(param_gamma[reco_mask_gamma]['width'], bins=width_bins, weights=param_gamma[reco_mask_gamma]['weights'])[0]
    proton_widthhist_gh = np.histogram(param_proton[reco_mask_proton]['width'], bins=width_bins, weights=param_proton[reco_mask_proton]['weights'])[0]

    # event rate as a function of energy after gh separation, theta2 cut
    fig = plt.figure()
    plt.step(mid_energy, param_proton_hist_nogh, 'r--', where='mid', label='reconstructed protons')
    plt.step(mid_energy, param_gamma_hist_nogh, 'b--', where='mid', label='reconstructed gammas')
    plt.step(mid_energy, param_proton_hist_gh_theta, 'r-', where='mid', label=r'protons after g/h sep. and $\Theta^2$ cuts')    # aplikovana vaha na pomer ploch oblasti
    plt.step(mid_energy, param_gamma_hist_gh_theta, 'b-', where='mid', label=r'gammas after g/h sep.  and $\Theta^2$ cuts')
    plt.ylabel('Event Rate [Hz]')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_protons_gammas_ghsep_theta2', dpi=100)
    plt.close(fig)

    # event rate as a function of theta2 after gh separation
    fig = plt.figure()
    plt.step(mid_theta, param_gamma_thetahist_gh, where='mid', label='gammas')
    plt.step(mid_theta, param_proton_thetahist_gh, where='mid', label='protons')
    plt.ylabel('Event Rate [Hz]')
    plt.xlabel('Theta2 [deg2]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('theta2_protons_gammas_ghsep.png', dpi=100)
    plt.close(fig)
    
    # event rate as a function of energy along the anlaysis 
    plt.figure(figsize=(8, 6))
    energy_bins_gamma_mid = .5 * (energy_bins_gamma[1:] + energy_bins_gamma[:-1])
    energy_bins_proton_mid = .5 * (energy_bins_proton[1:] + energy_bins_proton[:-1])
    #plt.step(energy_bins_proton_mid, param_proton_energy_hist_dl2, '--', label='p. reconstructed')
    plt.step(energy_bins_proton_mid, param_proton_energy_hist_qc, 'r-.', label='p. after quality cuts')
    plt.step(energy_bins_proton_mid, param_proton_energy_hist_gh, 'r:', label='p. after gh separation')
    plt.step(energy_bins_proton_mid, param_proton_energy_hist_reco, 'r-', label=r'p. after $\Theta^2$ cuts')
    #plt.step(energy_bins_gamma_mid, param_gamma_energy_hist_dl2, '--', label='reconstructed')
    plt.step(energy_bins_gamma_mid, param_gamma_energy_hist_qc, 'b-.', label=r'$\gamma$ after quality cuts')
    plt.step(energy_bins_gamma_mid, param_gamma_energy_hist_gh ,'b:', label=r'$\gamma$ after gh separation')
    plt.step(energy_bins_gamma_mid, param_gamma_energy_hist_reco, 'b-', label=r'$\gamma$ after $\Theta^2$ cuts')
    plt.ylabel('Event Rate [Hz]')
    plt.xlabel('$E_{reco}$ [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-2, 1e3])
    plt.ylim([1e-5, 1e2])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('rate_energy', dpi=100)
    plt.close(fig)
  

    fig = plt.figure()
    e_smooth = np.logspace(np.log10(0.001), np.log10(100), 100)
    plt.errorbar(mid_energy, mid_energy**2 * sensitivity_optimized , xerr=[err_low, err_up], color='red',  ecolor='red', elinewidth=2, fmt='-', markersize=0, label='LST mono')
    plt.plot(e_smooth, e_smooth**2 * crab_magic2015_dNdE(e_smooth), '-', alpha=0.5, color='grey', label='100% Crab')     # 100 %
    plt.plot(e_smooth, e_smooth**2 * 0.1 * crab_magic2015_dNdE(e_smooth), '--', alpha=0.5, color='grey', label='10% Crab')     # 10 %
    plt.plot(e_smooth, e_smooth**2 * 0.01 * crab_magic2015_dNdE(e_smooth), '-.', alpha=0.5, color='grey', label='1% Crab')     # 1 %
    plot_magic_sensitivity()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\mathsf{E^2 F \; [TeV \, m^{-2} s^{-1}]}$')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([10**-10, 10**-3])
    plt.legend()
    plt.tight_layout()
    plt.savefig('sensitivity_optimized.png', dpi=100)
    plt.close(fig)
    data_sensitiviy = np.stack([optimized_cuts[:, 0], optimized_cuts[:, 1], optimized_cuts[:, 2], sensitivity_optimized])
    np.savetxt('sensitivity_optimized.dat', data_sensitiviy.T, header='Emin_bin Emax_bin gammaness_cut sensitivity')

    fig = plt.figure()
    plt.hist2d(param_gamma_diffuse['offaxis_angle'],param_gamma_diffuse['error'], 100, norm=mpl.colors.LogNorm())
    plt.xlabel('off axis angle (deg)')
    plt.ylabel('source position error (deg)')
    cbar = plt.colorbar()
    cbar.set_label("# of entries")
    plt.savefig('error_function_offaxis.png', dpi=100)
    plt.close(fig)

    fig = plt.figure()
    plt.title('Intensity of excess events in signal area')
    intensity_bins_center = 0.5 * (intensity_bins[1:] + intensity_bins[:-1])
    plt.step(intensity_bins_center, proton_intensityhist_gh, 'r-', where='mid', label='proton')
    plt.step(intensity_bins_center, gamma_intensityhist_gh, 'b-', where='mid', label='gamma')
    plt.ylabel('rate [Hz]')
    plt.xlabel('Intensity [p.e.]')
    plt.xlim([10, 1e6])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('intensity_ghsep.png')
    plt.close(fig)

    fig = plt.figure()
    plt.title('Length of excess events in signal area')
    length_bins_center = 0.5 * (length_bins[1:] + length_bins[:-1])
    plt.step(length_bins_center, proton_lengthhist_gh, 'r-', where='mid', label='proton')
    plt.step(length_bins_center, gamma_lengthhist_gh, 'b-', where='mid', label='gamma')
    plt.ylabel('rate [Hz]')
    plt.xlabel('length [m]')
    plt.xlim([0, 1])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('length_ghsep.png')
    plt.close(fig)

    fig = plt.figure()
    plt.title('Width of excess events in signal area')
    width_bins_center = 0.5 * (width_bins[1:] + width_bins[:-1])
    plt.step(width_bins_center, proton_widthhist_gh, 'r-', where='mid', label='proton')
    plt.step(width_bins_center, gamma_widthhist_gh, 'b-', where='mid', label='gamma')
    plt.ylabel('rate [Hz]')
    plt.xlabel('Width [m]')
    plt.xlim([0, 0.2])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('width_ghsep.png')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0][0].set_title(r'gammas: intensity vs $\Theta^2$')
    param_gamma_theta2_intensity_2dhist_gh = np.histogram2d(
        param_gamma[mask_gh_gamma]['theta2'],
        param_gamma[mask_gh_gamma]['intensity'],
        bins=(theta_bins, intensity_bins),
    )[0]
    X, Y = np.meshgrid(theta_bins, intensity_bins)
    mesh = axes[0][0].pcolormesh(
        X, Y, param_gamma_theta2_intensity_2dhist_gh.T, 
        norm=mpl.colors.LogNorm()
    )
    plt.colorbar(mesh, ax=axes[0][0])
    axes[0][0].set_yscale('log')
    axes[0][0].set_xlabel(r'$\Theta^2$ [$\deg^2$]')
    axes[0][0].set_ylabel('intensity [pe]')
    axes[0][0].grid()
    axes[0][0].set_ylim([100, 1e5])

    axes[0][1].set_title(r'gammas: intensity vs $E_{true}$')
    #axes[0][1].set_title(r'gammas: intensity vs $E_{reco}$')
    param_gamma_theta2_energy_2dhist_gh = np.histogram2d(
        param_gamma[mask_gh_gamma]['mc_energy'],
        #param_gamma[mask_gh_gamma]['reco_energy'],
        param_gamma[mask_gh_gamma]['intensity'],
        bins=(energy_bins, intensity_bins),
    )[0]
    X, Y = np.meshgrid(energy_bins, intensity_bins)
    mesh = axes[0][1].pcolormesh(X, Y, param_gamma_theta2_energy_2dhist_gh.T, norm=mpl.colors.LogNorm())
    plt.colorbar(mesh, ax=axes[0][1])
    axes[0][1].set_xscale('log')
    axes[0][1].set_yscale('log')
    axes[0][1].set_xlabel(r'$E_{true}$ [TeV]')
    #axes[0][1].set_xlabel(r'$E_{reco}$ [TeV]')
    axes[0][1].set_ylabel('intensity [pe]')
    axes[0][1].grid()
    axes[0][1].set_ylim([100, 1e5])

    axes[1][0].set_title(r'protons: intensity vs length')
    param_proton_length_intensity_2dhist_gh = np.histogram2d(
        param_proton[mask_gh_proton]['length'],
        param_proton[mask_gh_proton]['intensity'],
        bins=(length_bins, intensity_bins),
    )[0]
    X, Y = np.meshgrid(length_bins, intensity_bins)
    mesh = axes[1][0].pcolormesh(
        X, Y, param_proton_length_intensity_2dhist_gh.T, 
        norm=mpl.colors.LogNorm()
    )
    plt.colorbar(mesh, ax=axes[1][0])
    axes[1][0].set_yscale('log')
    axes[1][0].set_xlabel(r'length [m]')
    axes[1][0].set_ylabel('intensity [pe]')
    axes[1][0].grid()
    axes[1][0].set_ylim([100, 1e5])

    axes[1][1].set_title(r'protons: intensity vs $E_{true}$')
    #axes[1][1].set_title(r'protons: intensity vs $E_{reco}$')
    param_proton_theta2_energy_2dhist_gh = np.histogram2d(
        param_proton[mask_gh_proton]['mc_energy'],
        #param_proton[mask_gh_proton]['reco_energy'],
        param_proton[mask_gh_proton]['intensity'],
        bins=(energy_bins, intensity_bins),
    )[0]
    X, Y = np.meshgrid(energy_bins, intensity_bins)
    mesh = axes[1][1].pcolormesh(X, Y, param_proton_theta2_energy_2dhist_gh.T, norm=mpl.colors.LogNorm())
    plt.colorbar(mesh, ax=axes[1][1])
    axes[1][1].set_xscale('log')
    axes[1][1].set_yscale('log')
    axes[1][1].set_xlabel(r'$E_{true}$ [TeV]')
    #axes[1][1].set_xlabel(r'$E_{reco}$ [TeV]')
    axes[1][1].set_ylabel('intensity [pe]')
    axes[1][1].grid()
    axes[1][1].set_ylim([100, 1e5])
    plt.tight_layout()
    plt.savefig('intensity_2dhisto.png')
    plt.close(fig)
