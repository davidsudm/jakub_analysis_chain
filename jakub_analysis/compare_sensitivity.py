#!/bin/env python
import numpy as np
from matplotlib import pyplot as plt


# Aleksic et al., 2015, JHEAP, (https://doi.org/10.1016/j.jheap.2015.01.002), fit od 0.1-20 TeV
def crab_magic2015_dNdE(energy):
    spectrum = 3.23 * 10**-11 * energy**(-2.47 - 0.24 * np.log10(energy))   # TeV-1 cm-2 s-1
    #spectrum = spectrum / 1.602176487   # erg-1 cm-2 s-1
    spectrum = spectrum * 10000 # TeV-1 m-2 s-1
    return spectrum


def plot_magic_sensitivity():
    magic_sens = np.array([0.068, 0.0367, 0.0225, 0.0203, 0.0151, 0.0153, 0.0166, 0.0236, 0.038, 0.061, 0.15])    # in % of Crab
    log10_energy_tev = [-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    energy = 10**np.array(log10_energy_tev)
    plt.plot(energy, energy**2 * magic_sens * crab_magic2015_dNdE(energy), 'k-', label='MAGIC stereo')


def compare_sensitivity(productions):
    fig = plt.figure()
    e_smooth = np.logspace(np.log10(0.001), np.log10(100), 100)
    plt.plot(e_smooth, e_smooth**2 * crab_magic2015_dNdE(e_smooth), '-', alpha=0.5, color='grey', label='100% Crab')     # 100 %
    plt.plot(e_smooth, e_smooth**2 * 0.1 * crab_magic2015_dNdE(e_smooth), '--', alpha=0.5, color='grey', label='10% Crab')     # 10 %
    plt.plot(e_smooth, e_smooth**2 * 0.01 * crab_magic2015_dNdE(e_smooth), '-.', alpha=0.5, color='grey', label='1% Crab')     # 1 %
    plot_magic_sensitivity()
    for prod in productions:
        filename = "/work/se2/yrenier/DL2/" + prod + "/jakub/sensitivity_optimized.dat"
        data = np.loadtxt(filename)
        ebin_min = data[:, 0]
        ebin_max = data[:, 1]
        gammaness_cut = data[:, 2]
        sensitivity = data[:, 3]
        mid_energy = (ebin_min + ebin_max)/2.0
        de_low = mid_energy - ebin_min
        de_up = ebin_max - mid_energy
        plt.errorbar(mid_energy, mid_energy**2 * sensitivity , xerr=[de_low, de_up], elinewidth=2, fmt='-', markersize=0, label=prod)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\mathsf{E^2 F \; [TeV \, m^{-2} s^{-1}]}$')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([10**-10, 10**-3])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('sensitivity_compare.png', dpi=100)
    plt.close(fig)


def compare_angular_resolution(productions):
    fig = plt.figure()
    for prod in productions:
        filename = "/work/se2/yrenier/DL2/" + prod + "/jakub/resolution_68.txt"
        data = np.loadtxt(filename)
        ebin_min = data[:, 0]
        ebin_max = data[:, 1]
        resolution_68 = data[:, 2]
        resolution_68[resolution_68 == 1] = np.nan
        mid_energy = (ebin_min + ebin_max)/2.0
        de_low = mid_energy - ebin_min
        de_up = ebin_max - mid_energy
        plt.errorbar(mid_energy, resolution_68 , xerr=[de_low, de_up], elinewidth=2, fmt='-', markersize=0, label=prod)
    plt.xscale('log')
    plt.ylabel('68% containement (deg)')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([0, 0.5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('angular_resolution_compare.png', dpi=100)
    plt.close(fig)


def compare_energy_resolution(productions):
    fig = plt.figure()
    for prod in productions:
        filename = "/work/se2/yrenier/DL2/" + prod + "/jakub/resolution_energy.txt"
        data = np.loadtxt(filename)
        ebin_min = data[:, 0]
        ebin_max = data[:, 1]
        resolution = data[:, 2]
        bias = data[:, 3]
        mid_energy = (ebin_min + ebin_max)/2.0
        de_low = mid_energy - ebin_min
        de_up = ebin_max - mid_energy
        plt.errorbar(mid_energy, resolution , xerr=[de_low, de_up], elinewidth=2, fmt='-', markersize=0, label=prod)
    plt.xscale('log')
    plt.ylabel('relative energy resolution (68%)')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([0, 0.5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_resolution_compare.png', dpi=100)
    plt.close(fig)


def compare_energy_bias(productions):
    fig = plt.figure()
    for prod in productions:
        filename = "/work/se2/yrenier/DL2/" + prod + "/jakub/resolution_energy.txt"
        data = np.loadtxt(filename)
        ebin_min = data[:, 0]
        ebin_max = data[:, 1]
        resolution = data[:, 2]
        bias = data[:, 3]
        mid_energy = (ebin_min + ebin_max)/2.0
        de_low = mid_energy - ebin_min
        de_up = ebin_max - mid_energy
        plt.errorbar(mid_energy, bias, xerr=[de_low, de_up], elinewidth=2, fmt='-', markersize=0, label=prod)
    plt.xscale('log')
    plt.ylabel('relative energy bias')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([-0.5, 0.5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_bias_compare.png', dpi=100)
    plt.close(fig)


def get_rate_rebined(filename, rebin=1):
    data = np.loadtxt(filename)
    ebin_min = data[:, 0]
    ebin_max = data[:, 1]
    rate = data[:, 2]
    nbin_rebin = np.ceil(ebin_min.size * (1./rebin) ).astype(int)
    ebin_min_rebin = np.zeros(nbin_rebin)
    ebin_max_rebin = np.zeros(nbin_rebin)
    rate_rebin = np.zeros(nbin_rebin)
    for i in range(nbin_rebin):
        index_bin_min = rebin * i
        ebin_min_rebin[i] = ebin_min[index_bin_min]
        index_bin_max = rebin*(i + 1) - 1
        if index_bin_max >= rate.size:
            index_bin_max = rate.size - 1
            rate_rebin[i] = np.sum(rate[index_bin_min:])
        else:
            rate_rebin[i] = np.sum(rate[index_bin_min:(index_bin_max+1)])
        ebin_max_rebin[i] = ebin_max[index_bin_max]
    return ebin_min_rebin, ebin_max_rebin, rate_rebin


def compare_rates(productions, rebin=1):
    fig = plt.figure()
    for prod in productions:
        filename_gamma = "/work/se2/yrenier/DL2/" + prod + "/jakub/rate_reco_gamma.txt"
        ebin_min_gamma, ebin_max_gamma, rate_gamma = get_rate_rebined(filename_gamma, rebin=rebin)
        mid_energy_gamma = (ebin_min_gamma + ebin_max_gamma)/2.0
        de_low = mid_energy_gamma - ebin_min_gamma
        de_up = ebin_max_gamma - mid_energy_gamma
        ebar_gamma = plt.errorbar(
            mid_energy_gamma, rate_gamma, xerr=[de_low, de_up], 
            elinewidth=2, fmt='.-', markersize=10, label=r'$\gamma$ ' + prod
        )
        col = ebar_gamma[0].get_color()
        filename_proton = "/work/se2/yrenier/DL2/" + prod + "/jakub/rate_reco_proton.txt"
        ebin_min_proton, ebin_max_proton, rate_proton = get_rate_rebined(filename_proton, rebin=rebin)
        mid_energy_proton = (ebin_min_proton + ebin_max_proton)/2.0
        de_low = mid_energy_proton - ebin_min_proton
        de_up = ebin_max_proton - mid_energy_proton
        plt.errorbar(
            mid_energy_proton, rate_proton, xerr=[de_low, de_up], 
            elinewidth=2, fmt='^--', color=col, markersize=5, label='proton ' + prod
        )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('rate [Hz}')
    plt.xlabel('E [TeV]')
    plt.xlim([0.01, 100])
    plt.ylim([1e-5, 0.3])
    plt.title(r'rates after g/h speration and $\Theta^2$ cut')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('rates_events_compare.png', dpi=100)
    plt.close(fig)    


if __name__ == '__main__':
    productions = ["LST-PMT-20deg", "LST-SiPM-20deg", "LST-SiPM-nofilter-20deg"]
    compare_rates(productions, rebin=4)
    compare_angular_resolution(productions)
    compare_sensitivity(productions)
    compare_energy_resolution(productions)
    compare_energy_bias(productions)

