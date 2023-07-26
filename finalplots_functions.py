## written by Sarah Betti 2022
# Fig 3 code written by Kate Follette 
## updated 26 July 2023 - commenting and cleanup

# functions for final plots for Betti+2022b

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib import cm, colors, transforms, gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle

from astropy.io import fits
from astropy import units as u
from astropy import constants as const

import warnings
warnings.simplefilter('ignore')

from scipy import spatial
from scipy.interpolate import interp1d

from specutils import Spectrum1D,SpectralRegion
from specutils.spectra import SpectralRegion
from specutils.manipulation import extract_region

import os
import urllib.request

from lmfit.models import VoigtModel, GaussianModel
from lmfit import report_fit
from specutils import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.fitting.continuum import fit_continuum
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines
from specutils.analysis import line_flux, equivalent_width

### FIGURE 1
def plot_features(ax, bound, wave, flux, features, feature_colors=None):
    """
    Take a list of *spectral features* and *colors* and place them above the location on the spectra.
    * Taken from SPLAT plot.py * 

    Parameters
    ----------
    ax : Axes
        The Axes to draw into.
    wave : list
        list of wavelength positions
    flux : list
        list of flux values
    features : list
        list of spectral lines
    feature_colors : list, optional
        list of feature colors.  If none, plot lines black. 
    bound : list
        bounds of plot
    """
    
    # create list of colors for each spectral lines.  each one will be plotted in black. 
    if feature_colors is None:
        feature_colors = ['k' for i in np.arange(len(features))]
        
    # dictionary of features. 
    feature_labels = { \
        'h2o': {'label': r'H$_2$O', 'type': 'band', 'wavelengths': [[0.925,0.95],[1.08,1.20],[1.72,1.9]]}, \
        'ch4': {'label': r'CH$_4$', 'type': 'band', 'wavelengths': [[1.1,1.2],[1.28,1.44],[1.6,1.76],[3.3, 3.45],[3.64,3.9]]}, \
        'co': {'label': r'CO', 'type': 'band', 'wavelengths': [[2.29,2.39]]}, \
        'tio': {'label': r'TiO', 'type': 'band', 'wavelengths': [[0.6569,0.6852],[0.705,0.727],[0.76,0.80],[0.825,0.831],[0.845,0.86]]}, \
        'vo': {'label': r'VO', 'type': 'band', 'wavelengths': [[1.04,1.08]]}, \
        'vo': {'label': r'VO', 'type': 'band', 'wavelengths': [[1.04,1.08]]}, \
        'young vo': {'label': r'VO', 'type': 'band', 'wavelengths': [[1.17,1.20]]}, \
        'cah': {'label': r'CaH', 'type': 'band', 'wavelengths': [[0.6346,0.639],[0.675,0.705]]}, \
        'crh': {'label': r'CrH', 'type': 'band', 'wavelengths': [[0.8611,0.8681]]}, \
        'feh': {'label': r'FeH', 'type': 'band', 'wavelengths': [[0.8692,0.875],[0.98,1.03],[1.19,1.25],[1.57,1.64]]}, \
        'sb': {'label': r'*', 'type': 'band', 'wavelengths': [[1.6,1.64]]}, \
        'paa': {'label': r'Pa$\alpha$', 'type':'line', 'wavelengths': [[1.874, 1.875]]}, \
        'pab': {'label': r'Pa$\beta$', 'type':'line', 'wavelengths': [[1.281, 1.282]]}, \
        'pag': {'label': r'Pa$\gamma$', 'type':'line', 'wavelengths': [[1.093, 1.094]]}, \
        'pad': {'label': r'Pa$\delta$', 'type':'line', 'wavelengths': [[1.004, 1.005]]}, \
        'bra': {'label': r'Br$\alpha$', 'type':'line', 'wavelengths': [[4.050, 4.051]]}, \
        'brb': {'label': r'Br$\beta$', 'type':'line', 'wavelengths': [[2.624, 2.625]]}, \
        'brg': {'label': r'Br$\gamma$', 'type':'line', 'wavelengths': [[2.166, 2.166]]}, \
        'brd': {'label': r'Br$\delta$', 'type':'line', 'wavelengths': [[1.943, 1.944]]}, \
        'br9': {'label': r'Br9', 'type':'line', 'wavelengths': [[1.816, 1.817]]}, \
        'br10': {'label': r'Br10', 'type':'line', 'wavelengths': [[1.457, 1.458]]}, \
        'pfb': {'label': r'Pf$\beta$', 'type':'line', 'wavelengths': [[4.654, 4.654]]}, \
        'pfg': {'label': r'Pf$\gamma$', 'type':'line', 'wavelengths': [[3.741, 3.741]]}, \
        'pfd': {'label': r'Pf$\delta$', 'type':'line', 'wavelengths': [[3.297, 3.297]]}, \
        'pf10': {'label': r'Pf10', 'type':'line', 'wavelengths': [[3.039, 3.039]]}, \
        'h': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]}, \
        'hi': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]}, \
        'h1': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]}, \
        'he1': {'label': r'He I', 'type': 'line', 'wavelengths': [[1.082, 1.083]]}, \
        'na': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137]]}, \
        'nai': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137]]}, \
        'na1': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137]]}, \
        'cs': {'label': r'Cs I', 'type': 'line', 'wavelengths': [[0.8521,0.8521],[0.8943,0.8943]]}, \
        'csi': {'label': r'Cs I', 'type': 'line', 'wavelengths': [[0.8521,0.8521],[0.8943,0.8943]]}, \
        'cs1': {'label': r'Cs I', 'type': 'line', 'wavelengths': [[0.8521,0.8521],[0.8943,0.8943]]}, \
        'rb': {'label': r'Rb I', 'type': 'line', 'wavelengths': [[0.78,0.78],[0.7948,0.7948]]}, \
        'rbi': {'label': r'Rb I', 'type': 'line', 'wavelengths': [[0.78,0.78],[0.7948,0.7948]]}, \
        'rb1': {'label': r'Rb I', 'type': 'line', 'wavelengths': [[0.78,0.78],[0.7948,0.7948]]}, \
        'mg': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'mgi': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'mg1': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'ca': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[0.6573,0.6573],[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'cai': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[0.6573,0.6573],[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'ca1': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[0.6573,0.6573],[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'caii': {'label': r'Ca II', 'type': 'line', 'wavelengths': [[1.184224,1.195301],[0.985746,0.993409]]}, \
        'ca2': {'label': r'Ca II', 'type': 'line', 'wavelengths': [[1.184224,1.195301],[0.985746,0.993409]]}, \
        'al': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511]]}, \
        'ali': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511]]}, \
        'al1': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511]]}, \
        'fe': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'fei': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'fe1': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'fe2': {'label': r'[Fe II]', 'type': 'band', 'wavelengths': [[1.256,1.256],[1.533,1.4],[1.643,1.643]]}, \
        'h2': {'label': r'H$_2$', 'type': 'line', 'wavelengths': [[1.12,1.13],[2.0337,2.0338],[2.1218,2.1218],[2.4065,2.4066]]}, \
        'k': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}, \
        'ki': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}, \
        'k1': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}}

    # interpolate over spectra
    nsamples = 1000
    f = interp1d(wave,flux,bounds_error=False,fill_value=0.)
    
    wvmax = np.arange(bound[0],bound[1],0.001)
    flxmax = f(wvmax)
         
    # find how far over spectra to plot line 
    yoff = 0.05*(bound[3]-bound[2])
    dbx = 0.01*(bound[1]-bound[0])
    
    # fonsize.  should probably be automatic or kwarg...
    fontsize = 14
    
    # loop through each feature and plot
    for J, ftr in enumerate(features):
        ftr = ftr.lower()
        if ftr in feature_labels:
            for ii,waveRng in enumerate(feature_labels[ftr]['wavelengths']):
                wRng = waveRng
                if (np.min(wRng) > bound[0] and np.max(wRng) < bound[1]):
                    dwrng = np.nanmax(wRng)-np.nanmin(wRng)
                    if dwrng < dbx: dwrng = dbx
                    x = (np.arange(0,nsamples+1.0)/nsamples)*dwrng*1.1+np.nanmin(wRng)-dwrng*0.05
                    wfeature = np.where(np.logical_and(wvmax >= x[0],wvmax <= x[-1]))
                    f = interp1d(np.array(wvmax)[wfeature],np.array(flxmax)[wfeature],bounds_error=False,fill_value=0.)
                    y = np.nanmax(f(x))+1.*yoff

                    if feature_labels[ftr]['type'] == 'band':
                        ax.plot(wRng,[y+yoff]*2,color=feature_colors[J],linestyle='-', zorder=10)
                        ax.plot([wRng[0]]*2,[y,y+yoff],color=feature_colors[J],linestyle='-', zorder=10)
                        ax.text(np.mean(wRng),y+1.5*yoff,feature_labels[ftr]['label'],horizontalalignment='center',fontsize=fontsize, zorder=10, color=feature_colors[J])
                    else:
                        for w in wRng: ax.plot([w]*2,[y,y+yoff],color=feature_colors[J],linestyle='-')
                        ax.text(np.mean(wRng),y+1.5*yoff,feature_labels[ftr]['label'],horizontalalignment='center',fontsize=fontsize,zorder=10,color=feature_colors[J])
                        #wRng = [wRng[0]-0.02,wRng[1]+0.02]   # for overlap

                    foff = np.zeros(len(flxmax))
                    foff[wfeature] = 3.*yoff
                    flxmax = list(np.array(flxmax)+foff)
    bound[3] = np.nanmax([np.nanmax(flxmax)+2.*yoff,bound[3]])
    
def transform_to_velocity(data,centroid, **kwargs):
    """transform from wavelength to velocity space
    
    input
    --------
    data       : (astropy fits table) 
    centroid   : (the zero velocity point) in um, same as wavelengths
    
    returns
    --------
    velocity_array : the wavelength values transformed to velocities relative to the chosen centroid
    
    """
    if kwargs.get('not_soarapo'):
        data = data.data[0]
        wave = data[0]/1000.
    else:
        data = data.data
        wave = data[0]

    c = 2.99e5 # speed of light in km/s
    
    
    delta_lambda = (wave - centroid)/centroid    # the fractional wavelength difference
    velocity_array = c * delta_lambda

    return velocity_array # this is in km/s, as defined above
    
    
def mods(data):
  
    X = data.spectral_axis.to(u.AA)
    data = Spectrum1D(data.flux, spectral_axis=X)
   
    mid = data.spectral_axis.value[np.argmax(data.flux.value)]
    region_cont = [(data.spectral_axis.min(), (mid-(10))*u.AA), 
               ((mid+(10))*u.AA, data.spectral_axis.max())]
   
    fitted_continuum = fit_continuum(data, window=region_cont)
    y_fit = fitted_continuum(data.spectral_axis)
    cont = np.nanmean(y_fit) 
    spec_normalized = data - cont
    newx= np.linspace(spec_normalized.spectral_axis.min(), spec_normalized.spectral_axis.max(),200)
    Gaussian_model = GaussianModel()
    params = Gaussian_model.guess(spec_normalized.flux.value, x=spec_normalized.spectral_axis.value)
    params['sigma'].set(value=1.5, vary=True, expr='', min=0, max=5) # force gamma to be > 0
    params['center'].set(value=mid, vary=True, expr='', min=mid-(15), max=mid+(15)) # force gamma to be > 0
    data_bestfit  = Gaussian_model.fit(spec_normalized.flux.value, params, 
                                   x=data.spectral_axis.value)
    ynew = Gaussian_model.eval(data_bestfit.params, x=newx.value)
    return newx.to(u.um), ynew+cont.value, cont.value

def plot_pretty(ax, Del_final, Del_final2, C1, C2, bound, features = None, feature_colors=None, accretors_list=None,  feats=False, **kwargs):
    
    '''
    Parameters
    -------
    ax : matplotlib axis
        axis for plotting
    Del_final : astropy.fits 
        spectra 1 in astropy.fits format (fits.open()[0])
    Del_final2 : astropy.fit
        spectra 2 in astropy.fits format (fits.open()[0])
    C1 : str
        color for spectra 1
    C2 : str
        color for spectra 2
    bound : list
        [xlim_min, xlim_max, ylim_min, ylim_max]
        limits for x and y axes
    features : list
        spectral features to highlight on plot
    feature_colors : list
        color of spectral features to highlight on plot
    accretor_list : list
        list of accretors to highlight
    feats : boolean 
        boolean of whether or not to show accretors
        if True, must have featurs, feature_colors, accretor_list
   
    **kwargs
        t : float
            y position of line label 
        h : float
            y position of inset axes
            
        label : list
            list of legend labels
        fs : float
            fontsize for axis labels
        
    '''
    label = kwargs.get('label', ['spectra 1', 'spectra 2'])
    t = kwargs.get('t', 0.8)
    h = kwargs.get('h', 0.48)
    fs = kwargs.get('fs', 18)
    
    # plot epoch 1
    sp_real_wave_X = Del_final.data[0] 
    sp_real_flux_Y_norm = Del_final.data[1] / np.nanmean(Del_final.data[1])
    ax.plot(sp_real_wave_X, sp_real_flux_Y_norm,linewidth=2, color=C1,label=label[0])
    
    # plot epoch 2
    DelABbX2 = Del_final2.data[0] 
    DelABbY2 = Del_final2.data[1] / np.nanmean(Del_final2.data[1])
    ax.plot(DelABbX2, DelABbY2,linewidth=1., color=C2, label=label[1], zorder=10, alpha=1)
    
    # make legend
    leg = ax.legend(loc='lower left', fontsize=16, facecolor='none')
    for line in leg.get_lines():
        line.set_linewidth(3)
        
    # plot the features on to topmost spectra
    if feats:
        plot_features(ax, bound, sp_real_wave_X, sp_real_flux_Y_norm, features, feature_colors=feature_colors)

#     plot the gray markers.  the left and right x limits are found by hand using the transmission plot. 
#     LR = rightside, LL = left side.  
    LR = [1.113, 1.333, 1.764]
    LL = [1.156, 1.491, 2.0]
    for P in np.arange(len(LR)):
        ylim_min = bound[2]
        H, WW = 5 - ylim_min, LL[P] - LR[P]
        if P == 0:
            RECT = patches.Rectangle((LR[P],ylim_min),WW,H,linewidth=1,edgecolor='gainsboro',
                              facecolor='gainsboro', alpha=0.3, label='atmospheric transmission < 0.3')
        else:
            RECT = patches.Rectangle((LR[P],ylim_min),WW,H,linewidth=1,edgecolor='gainsboro',
                      facecolor='gainsboro', alpha=0.3)
        ax.add_patch(RECT)
        
    # add inset axes
    ax4 = ax.inset_axes([.83, h, .15, 0.4])
    ax3 = ax.inset_axes([.66, h, .15, 0.4])
    ax2 = ax.inset_axes([.49, h, .15, 0.4])
    
    units = u.W * u.m**-2 * u.um**-1
    spectrum = Spectrum1D(flux=sp_real_flux_Y_norm*units, spectral_axis = sp_real_wave_X*u.um)
    spectrum2 = Spectrum1D(flux=DelABbY2*units, spectral_axis = DelABbX2*u.um)

    region = SpectralRegion(1.279* u.um, 1.285 * u.um)
    sub_spectrum2_PaB = extract_region(spectrum, region)
    sub_spectrum2_PaB2 = extract_region(spectrum2, region)

    region = SpectralRegion(1.0915* u.um, 1.0965 * u.um)
    sub_spectrum2_PaG = extract_region(spectrum, region)
    sub_spectrum2_PaG2 = extract_region(spectrum2, region)

    region = SpectralRegion(2.154* u.um, 2.178 * u.um)
    sub_spectrum2_BrG = extract_region(spectrum, region)
    sub_spectrum2_BrG2 = extract_region(spectrum2, region)

    c = 2.99e5 # speed of light in km/s
    if feats:
        # fit PaB emission line
        for mod in [sub_spectrum2_PaB2, sub_spectrum2_PaB]:
            XX, YY, cont = mods(mod)
            ax2.plot(XX, YY, color='k', zorder=100, linestyle='--', linewidth=1.5)
            
    # plot inset axis for PaB
    ax2.plot(sub_spectrum2_PaB2.spectral_axis, sub_spectrum2_PaB2.flux.value, color=C2, zorder=10, linewidth=3., )
    ax2.plot(sub_spectrum2_PaB.spectral_axis, sub_spectrum2_PaB.flux, color=C1, linewidth=3) 
    ax2.text(0.1, kwargs['t'], 'Paβ', transform=ax2.transAxes, fontsize=15)
    ax2.set_ylim(top = 0.35+np.nanmax(sub_spectrum2_PaB.flux.value))

    delta_lambda = (sub_spectrum2_PaB2.spectral_axis.value - 1.282)/1.282    # the fractional wavelength difference
    velocity_array = c * delta_lambda
    ax2t = ax2.twiny()
    min_x, max_x = velocity_array[0], velocity_array[-1]
    ax2t.set_xlim([min_x, max_x])
    
    if feats:
        # fit PaG emission line
        for mod in [sub_spectrum2_PaG2, sub_spectrum2_PaG]: #
            XX, YY, cont = mods(mod)
            ax3.plot(XX, YY, color='k', zorder=100, linestyle='--', linewidth=1.5)
    # plot inset axis for PaG
    ax3.plot(sub_spectrum2_PaG2.spectral_axis, sub_spectrum2_PaG2.flux.value, color=C2, zorder=10, linewidth=3., ) 
    ax3.plot(sub_spectrum2_PaG.spectral_axis, sub_spectrum2_PaG.flux, color=C1, linewidth=3) 
    ax3.text(0.1, kwargs['t'], 'Paγ', transform=ax3.transAxes, fontsize=15)
    ax3.set_ylim(top = 0.35+np.nanmax(sub_spectrum2_PaG2.flux.value))
    delta_lambda = (sub_spectrum2_PaG2.spectral_axis.value - 1.094)/1.094    # the fractional wavelength difference
    velocity_array = c * delta_lambda
    ax3t = ax3.twiny()
    min_x, max_x = velocity_array[0], velocity_array[-1]
    ax3t.set_xlim([min_x, max_x])
    
    if feats:
        # fit PaG emission line
        XX, YY, cont = mods(sub_spectrum2_BrG)
        ax4.plot(XX, YY, color='k', zorder=100, linestyle='--', linewidth=1.5)
    
    # plot inset axis for BrG
    ax4.plot(sub_spectrum2_BrG2.spectral_axis, sub_spectrum2_BrG2.flux.value, color=C2,zorder=10,linewidth=3., alpha=1) 
    ax4.plot(sub_spectrum2_BrG.spectral_axis, sub_spectrum2_BrG.flux, color=C1, linewidth=3) 
    ax4.text(0.1, kwargs['t'], 'Brγ', transform=ax4.transAxes, fontsize=15)
    ax4.set_ylim(top = 0.2+np.nanmax(sub_spectrum2_BrG2.flux.value))
    delta_lambda = (sub_spectrum2_BrG2.spectral_axis.value - 2.166)/2.166    # the fractional wavelength difference
    velocity_array = c * delta_lambda
    ax4t = ax4.twiny()
    min_x, max_x = velocity_array[0], velocity_array[-1]
    ax4t.set_xlim([min_x, max_x])
    
    for axi in [ax2, ax3, ax4]:
        axi.tick_params(top=False, right=True, labelleft=True, direction='in' )
        axi.set_xlabel('λ (μm)')
#         axi.set_facecolor('none')
        
    for axi in [ax2t, ax3t, ax4t]:
        axi.tick_params(top=True, bottom =False, right=True, labelleft=False, direction='in' )
        axi.set_xlabel('v (km/s)')
#         axi.set_facecolor('none')
        
    ax.tick_params(which='major', top=True, bottom=True, right=True, length=10,direction='in', labelsize=fs, width=1)
    ax.tick_params(which='minor', top=True, bottom=True, right=True, length=7,direction='in', labelsize=fs, width=1)
    ax.minorticks_on()
    xlim_min, xlim_max,ylim_min, ylim_max = bound
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([ylim_min, ylim_max])
    
    
############ FIGURE 3

def read_kf11(fstr):
    temps=['3750','5000','7500','8750','10000','12500','15000']
    colnames = ['log n','3750','5000','7500','8750','10000','12500','15000']
    #read in first column (logn) as index
    line_data = pd.read_csv('KF11_CaseI/'+fstr+'_pabeta.txt', delim_whitespace=True, skiprows = 2, names=colnames, index_col=0)
    return(line_data)

#little function to truncate colormap and get rid of overlapping colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def model_stuff(nvals, trans_range, amin=0.4):
    crange = np.arange(len(nvals))/len(nvals)

    #define transparency ranges for second axis of colorbar
    #aoyama vel
#     amin=0.2 #minimum transparency
    arange = (np.arange(len(trans_range))+1)*(1 - amin)/(len(trans_range))+amin

    #populate color and transparency lists
    cb1 = np.zeros((len(nvals),len(trans_range)))
    a1 = np.zeros((len(nvals),len(trans_range)))
    for i in np.arange(len(nvals)):
        cb1[i,:]=crange[i]
    for j in np.arange(len(trans_range)):
        a1[:,j]=arange[j]
    return crange, arange, cb1, a1



def ratio_plot_4panel(aoyama, l1, l2, CMAP1, CMAP2, highlight=False, uncert=False):
    # crazy plot ...
    ao_v0s = np.unique(aoyama['v0'])
    ao_nvals = np.unique(aoyama['logn'])
    kf_nvals = np.arange(80,126,2)/10.
    kf_temps=['3750','5000','7500','8750','10000','12500','15000']
    linedict={'Ha':'3-2','PaA':'4-3','PaG':'6-3','PaD':'7-3','BrA':'5-4','BrG':'7-4'}

    delflux1={'Ha':12.8e-16, 'PaB':8.05e-16,'PaG':6.825e-16,'BrG':1.64e-16}
    delstdev1={'Ha':0.7e-16, 'PaB':0.56e-16,'PaG':0.614e-16,'BrG':0.29e-16}
    
    delflux2={'Ha':12.8e-16, 'PaB':3.49e-16,'PaG':2.94e-16,'BrG':6.4e-17}
    delstdev2={'Ha':0.7e-16, 'PaB':0.52e-16,'PaG':0.62e-16,'BrG':np.nan}
    
    wid_ratios = np.repeat(2,len(l1)+2)
    wid_ratios[-2:]=1.75
    
    # make figure
    f = plt.figure(figsize=(15,9))
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[1,1], figure=f, 
                            right=0.7)
    gs01 = gridspec.GridSpec(ncols=1, nrows=2, figure=f, 
                            left=0.72, hspace=0.4)

    ax2 = f.add_subplot(gs01[0,0])
    ax3 = f.add_subplot(gs01[1,0])
    f.set_facecolor('white')

    # loop through each line ratio pair / subplot
    k=0
    let = ['a)', 'b)', 'c)', 'd)']
    for k in np.arange(len(l1)+2):
        map1 = sns.color_palette(CMAP1, as_cmap=True)
        map2 = sns.color_palette(CMAP2, as_cmap=True)
        
        crange1, arange1, cb1, a1 = model_stuff(ao_nvals, ao_v0s, amin=0.4)
        crange2, arange2, cb2, a2 = model_stuff(kf_nvals, kf_temps, amin=0.2)
        
        
        if k<np.arange(len(l1)+2)[-2]:
            # read in K&F 11
            r1 = l1[k].split('/')
            if r1[1] != 'PaB':
                #bootstrap non-PaB ratios
                l1_num = read_kf11(linedict[r1[0]])
                l1_denom = read_kf11(linedict[r1[1]])
                l1_kf = l1_num/l1_denom
            else:  
                #PaB
                l1_kf = read_kf11(linedict[r1[0]])
      
            #same for second axis
            r2 = l2[k].split('/')
            if r2[1] != 'PaB':
                l2_num = read_kf11(linedict[r2[0]])
                l2_denom = read_kf11(linedict[r2[1]])
                l2_kf = l2_num/l2_denom
            else:  
                l2_kf = read_kf11(linedict[r2[0]])

            i=0
            ax = f.add_subplot(spec[k])
            ax.text(0.05, 0.85, let[k], transform=ax.transAxes, fontsize=24)
            
            # PLOT K&F 11
            for n in kf_nvals:
                for j in np.arange(len(arange2)):
                    ax.plot(np.log10(l1_kf[l1_kf.index==n].values[0][j]), 
                            np.log10(l2_kf[l2_kf.index==n].values[0][j]),
                  's', color=map2(crange2[i]), markersize=12, alpha=arange2[j])
                i+=1
            
            # PLOT AOYAMA 2018
            i=0
            vs = []
            ns = []
            xs = []
            ys = []
            
            for n in ao_nvals:
                for j in np.arange(len(ao_v0s)):
                    ao_subarr = aoyama[aoyama['logn']==n][aoyama['v0']==ao_v0s[j]]
                    if (str(r1[0]) in ao_subarr.keys()) & (str(r1[1]) in ao_subarr.keys()) & (str(r2[0]) in ao_subarr.keys()) & (str(r2[1]) in ao_subarr.keys()):
                        l1_ao =np.log10(ao_subarr[str(r1[0])]/ao_subarr[str(r1[1])])
                        l2_ao = np.log10(ao_subarr[str(r2[0])]/ao_subarr[str(r2[1])]) 
                        #print(l1_ao, l2_ao)
                        ax.plot(l1_ao,l2_ao,'o',color=map1(crange1[i]), markersize=12,
                                alpha=arange1[j], mec='k', mew=1) #,label='A20, n='+str(n))
                        vs.append((np.ones_like(l1_ao)*j)[0])
                        ns.append((np.ones_like(l1_ao)*i)[0])
                        xs.append(l1_ao.values[0])
                        ys.append(l2_ao.values[0])
                i+=1
        
            xs = np.array(xs)
            ys = np.array(ys)
            ns = np.array(ns)
            vs = np.array(vs)

            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

            # PLOT OBSERVED LINE RATIOS
            if highlight != False:  
                # CALCULATE LINE RATIO 
                
                ################ EPOCH 1 ################
                XX = np.log10(delflux1[r1[0]]/delflux1[r1[1]])
                YY = np.log10(delflux1[r2[0]]/delflux1[r2[1]])

                
                errXX = 0.434 *  np.sqrt( (delstdev1[r1[0]]/delflux1[r1[0]])**2.+ (delstdev1[r1[1]]/delflux1[r1[1]])**2.   )
                errYY = 0.434 * np.sqrt( (delstdev1[r2[0]]/delflux1[r2[0]])**2.+ (delstdev1[r2[1]]/delflux1[r2[1]])**2.   )
                
                if 'Ha' in r1[1]:
                    color='white'
                    mec='k'
                    ms=24
                    barsabove=True
                else:
                    color='k'
                    mec='k'
                    ms=24
                    barsabove=True
                ax.errorbar(XX, YY, xerr= errXX, yerr = errYY,
                       marker='*', color=color, markeredgecolor=mec, markeredgewidth=1, markersize=ms, 
                           elinewidth=1.5, ecolor='k', capsize=3, zorder=120,barsabove=barsabove) 
                ax.errorbar(XX, YY, xerr= errXX, yerr = errYY,
                       marker='*', color='white', markeredgecolor='white', markeredgewidth=3, markersize=ms+10, 
                           elinewidth=4, ecolor='white', capsize=5, zorder=100,barsabove=barsabove) 
                if (str(r1[0]) in aoyama.keys()) & (str(r1[1]) in aoyama.keys()) & (str(r2[0]) in aoyama.keys()) & (str(r2[1]) in aoyama.keys()):
                    import matplotlib.patches as patches

                    # The ellipse
                    g_ell_center = (XX, YY)
                    g_ell_width  = errXX
                    g_ell_height = errYY
                    angle = 0.

                    cos_angle = np.cos(np.radians(180.-angle))
                    sin_angle = np.sin(np.radians(180.-angle))

                    xc = xs - g_ell_center[0]
                    yc = ys - g_ell_center[1]

                    xct = xc * cos_angle - yc * sin_angle
                    yct = xc * sin_angle + yc * cos_angle 

                    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

                    # Set the colors. Black if outside the ellipse, green if inside
                    xxs = np.array(xs[np.where(rad_cc <= 1.)[0]])
                    yys = np.array(ys[np.where(rad_cc <= 1.)[0]])
                    vv = np.array(vs[np.where(rad_cc <= 1.)[0]])
                    nn = np.array(ns[np.where(rad_cc <= 1.)[0]])
                    if len(vv) > 0:
                        meanvv = np.nanmean(vv)
                        meannn = np.nanmean(nn)
                        stdvv = np.nanstd(vv)
                        stdnn = np.nanstd(nn)
                        ax3.errorbar(meanvv,meannn, xerr=stdvv, yerr=stdnn, marker='*', ms=24,mew=3,mec='k',mfc='k',ecolor='k', elinewidth=3, capsize=3, zorder=100 )

                ################ EPOCH 2 ################
                
                XX = np.log10(delflux2[r1[0]]/delflux2[r1[1]])
                YY = np.log10(delflux2[r2[0]]/delflux2[r2[1]])
                errXX = 0.434 * np.sqrt( (delstdev2[r1[0]]/delflux2[r1[0]])**2. + (delstdev2[r1[1]]/delflux2[r1[1]])**2.   )
                errYY = 0.434 * np.sqrt( (delstdev2[r2[0]]/delflux2[r2[0]])**2. + (delstdev2[r2[1]]/delflux2[r2[1]])**2.   )
                if (np.isnan(errXX)) | (np.isnan(errYY)):
                    if np.isnan(errXX):
                        xuplims=True
                        if k == 2:
                            errXX = 0.02
                        else:
                            errXX = 0.1
                    else:
                        xuplims=False
                    if np.isnan(errYY):
                        uplims=True
                        
                        if k == 1:
                            errYY=0.05
                        else:
                            errYY=0.1
                        
                    else:
                        uplims=False
                    if 'Ha' in r1[1]:
                        color='white'
                        mec='k'
                        ms=14
                        barsabove=True
                        
                    else:
                        color='k'
                        mec='k'
                        ms=13
                        barsabove=True
                    ax.errorbar(XX, YY, xerr= errXX, yerr = errYY, xuplims=xuplims, uplims=uplims,
                        marker='D', color=color, markeredgecolor=mec, markeredgewidth=1, markersize=ms, 
                               elinewidth=1.5, ecolor='k', capsize=3, zorder=110, barsabove=barsabove) 
                    ax.errorbar(XX, YY, xerr= errXX, yerr = errYY, xuplims=xuplims, uplims=uplims,
                        marker='D', color='white', markeredgecolor='white', markeredgewidth=3, markersize=ms+2, 
                               elinewidth=4, ecolor='white', capsize=5, zorder=100, barsabove=barsabove) 
                else: 
                    if 'Ha' in r1[1]:
                        color='white'
                        mec='k'
                        ms=14
                        barsabove=True
                    else:
                        color='k'
                        mec='k'
                        ms=13
                        barsabove=True
                    ax.errorbar(XX, YY, xerr= errXX, yerr = errYY,
                           marker='D', color=color, markeredgecolor=mec, markeredgewidth=1, markersize=ms, 
                               elinewidth=1.5, ecolor='k', capsize=3, zorder=110, barsabove=barsabove) 
                    ax.errorbar(XX, YY, xerr= errXX, yerr = errYY, xuplims=xuplims, uplims=uplims,
                        marker='D', color='white', markeredgecolor='white', markeredgewidth=3, markersize=ms+2, 
                               elinewidth=4, ecolor='white', capsize=5, zorder=100, barsabove=barsabove) 
                if (str(r1[0]) in aoyama.keys()) & (str(r1[1]) in aoyama.keys()) & (str(r2[0]) in aoyama.keys()) & (str(r2[1]) in aoyama.keys()):
                    import matplotlib.patches as patches

                    # The ellipse
                    g_ell_center = (XX, YY)
                    if (errXX == 0.1) & ((errYY == 0.1) or (errYY == 0.05) or (errXX == 0.02)):
                        g_ell_width=0.05
                        g_ell_height= 0.05
                        
                    elif (errXX != 0.1) & ((errYY == 0.1) or (errYY == 0.05)or (errXX != 0.02)):
                        g_ell_width=errXX
                        g_ell_height= 0.05
                    elif (errXX == 0.1) & ((errYY != 0.1) or (errYY != 0.05)or (errXX == 0.02)):
                        g_ell_width=0.05
                        g_ell_height= errYY
                    else:
                        g_ell_width = errXX
                        g_ell_height = errYY
                    angle = 0.

                    cos_angle = np.cos(np.radians(180.-angle))
                    sin_angle = np.sin(np.radians(180.-angle))

                    xc = xs - g_ell_center[0]
                    yc = ys - g_ell_center[1]

                    xct = xc * cos_angle - yc * sin_angle
                    yct = xc * sin_angle + yc * cos_angle 

                    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

                    # Set the colors. Black if outside the ellipse, green if inside
                    xxs = np.array(xs[np.where(rad_cc <= 1.)[0]])
                    yys = np.array(ys[np.where(rad_cc <= 1.)[0]])
                    vv = np.array(vs[np.where(rad_cc <= 1.)[0]])
                    nn = np.array(ns[np.where(rad_cc <= 1.)[0]])
                    if (errXX == 0.1) & ((errYY == 0.1) or (errYY == 0.05) or (errXX == 0.02)):
                        xxs2 = xxs[np.where((xxs <= XX) & (yys <= YY))[0]]
                        yys2 = yys[np.where((xxs <= XX) & (yys <= YY))[0]]
                        vv2 = vv[np.where((xxs <= XX) & (yys <= YY))[0]]
                        nn2 = nn[np.where((xxs <= XX) & (yys <= YY))[0]]
                    elif (errXX != 0.1) & ((errYY == 0.1) or (errYY == 0.05)or (errXX != 0.02)):
                        xxs2 = xxs[np.where(yys<=YY)[0]]
                        yys2 = yys[np.where(yys<=YY)[0]]
                        vv2 = vv[np.where(yys<=YY)[0]]
                        nn2 = nn[np.where(yys<=YY)[0]]
                    elif (errXX == 0.1) & ((errYY != 0.1) or (errYY != 0.05)or (errXX == 0.02)):
                        xxs2 = xxs[np.where(xxs <= XX)[0]]
                        yys2 = yys[np.where(xxs <= XX)[0]]
                        vv2 = vv[np.where(xxs <= XX)[0]]
                        nn2 = nn[np.where(xxs <= XX)[0]]
                    else:
                        xxs2 = xxs
                        yys2 = yys
                        vv2 = vv
                        nn2 = nn
                        
                    if len(vv2) > 0:
                        meanvv = np.nanmean(vv2)
                        meannn = np.nanmean(nn2)
                        stdvv = np.nanstd(vv2)
                        stdnn = np.nanstd(nn2)
                        print(meanvv, meannn)
                        ax3.errorbar(meanvv,meannn, xerr=stdvv, yerr=stdnn, marker='D', ms=12,mew=3,mec='k',mfc='k',ecolor='k', elinewidth=3, capsize=3, zorder=100 )


            ax.set_xlabel('log '+ l1[k], fontsize=16)
            ax.set_ylabel('log '+l2[k], fontsize=16)
            #replace ABGD with greek letters in axis labels
            greekletters={'PaA':'Pa$\\alpha$','PaB':'Pa$\\beta$','PaG':'Pa$\gamma$','PaD':'Pa$\delta$','BrA':'Br$\\alpha$','BrG':'Br$\gamma$', 'Ha':'H$\\alpha$'}
            xlab = l1[k]
            ylab = l2[k]
            print(xlab)
            for letter in greekletters:
                xlab=xlab.replace(letter, greekletters[letter])
                ylab=ylab.replace(letter, greekletters[letter])

            #label axes and set tick lengths, widths
            ax.set_xlabel('log '+ xlab, fontsize=16)
            ax.set_ylabel('log '+ ylab, fontsize=16)
            ax.tick_params(which='major', length=10, width=2, direction='in', top=True, right=True, labelsize=16)
            ax.tick_params(which='minor', length=6, width=2, direction='in', top=True, right=True, labelsize=16)
            ax.minorticks_on()
            if k==0:
                ax.set_ylim(-1.4, -0.2)

        # make colorbars
        elif k==np.arange(len(l1)+2)[-2]:
            ax2.imshow(cb2, vmax=1.11, cmap=map2,alpha=a2, origin='lower')
            ax2.set_title('Local Line Excit. Models\n (Kwan & Fischer 2011)', fontsize=14)
            ax2.set_xlabel('Temperature (10$^3$ K)', fontsize=14)
            ax2.set_ylabel('log(in situ $n$) (cm$^{-3}$)', fontsize=14)
            #make (labeled) major ticks every 5 grid values in y (number dens)
            ax2.set_yticks(np.arange(len(kf_nvals))[::5])
            ax2.set_yticklabels(kf_nvals[::5])
            #label every temp, but in units of 10^3 so fit
            ax2.set_xticks(np.arange(len(kf_temps)))
            kf_temp_labels = [float(i)/1000. for i in kf_temps]
            ax2.set_xticklabels(['3.75','5','7.5','8.75','10','12.5','15'])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
                        #       ax2.xaxis.set_minor_locator(AutoMinorLocator())
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
            ax2.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
            ax2.set_aspect(0.42)
        else:
            ax3.imshow(cb1, vmax=1.05, cmap=map1,alpha=a1, origin='lower', zorder=10)
            ax3.set_title('Planetary Shock Models\n(Aoyama et al. 2018)', fontsize=14)
            ax3.set_xlabel('Preshock $v$ (km/s)', fontsize=14)
            ax3.set_ylabel('log(preshock $n$) (cm$^{-3}$)', fontsize=14)
            ax3.set_xticks(np.arange(len(ao_v0s))[::4])
            ax3.set_xticklabels(ao_v0s[::4])
            ax3.set_yticks(np.arange(len(ao_nvals)))
            ax3.set_yticklabels(['9','10','11','12','13','14'])

            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax3.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
            ax3.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
            ax3.set_aspect(4.38)
        print()
    k+=1 
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, hspace=0.25, wspace=0.25, right=0.97)
    plt.savefig('/Users/sarah/Desktop/fig3.pdf', dpi=150, transparent=True)
