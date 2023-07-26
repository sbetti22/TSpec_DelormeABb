## written by Sarah Betti 2022
## updated 26 July 2023 - commenting and cleanup

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import constants as const

import warnings; warnings.simplefilter('ignore')
from astropy import log
log.setLevel('ERROR')

from matplotlib import pyplot as plt

from specutils import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.fitting.continuum import fit_continuum
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines
from specutils.analysis import line_flux, equivalent_width

from scipy.stats import norm
from lmfit.models import VoigtModel, GaussianModel
from lmfit import report_fit

import pandas as pd

import extinction
from extinction import remove, ccm89

from itertools import groupby
from operator import itemgetter
import more_itertools as mit
from scipy import interpolate

class measure_Mdot:
    def __init__(self, wavelength, flux, flux_err, Av, xunit='um', yunit='W/m2/um'):
        '''
        Purpose 
        ----------
        measure line fluxes -> accretion rates for NIR data from SOAR for PaB, PaG, and BrG lines
        
        Parameters
        ----------
        wavelength : 1d numpy.ndarray
            wavelength data 
        flux : 1d numpy.ndarray
            flux data 
        flux_err : 1d numpy.ndarray
            flux error data in W/m2/um
        Av : float
            visual extinction
        xunit : str (optional)
            unit of wavelength array
        yunit : str (optional)
            unit of flux array
        '''
        self.Av=Av
        # convert wavelength to A
        self.wavelength = (np.array(wavelength) * u.Unit(xunit)).to(u.AA)
        WW = self.wavelength.value.astype('double')
        # deredden using CCM89 with R =3.1
        flux = remove(ccm89(WW, Av, 3.1), flux)
        
        # convert flux to erg/s/cm2/A
        self.flux = (flux * u.Unit(yunit)).to(u.erg / u.s/ u.cm**2/u.AA)
        self.flux_err = StdDevUncertainty((flux_err * u.Unit(yunit)).to(u.erg / u.s/ u.cm**2/u.AA))
        # create specutils Spectrum1D object
        self.spec = Spectrum1D(spectral_axis = self.wavelength, flux=self.flux, uncertainty=self.flux_err)
        self.xunit = xunit
        self.yunit = yunit

    def hydrogen_lines(self, line):
        '''
        return wavelength in Angstrom of specific hydrogen lines
        '''
        hydrogen = {}
        hydrogen['BrG'] = 21661.2
        hydrogen['PaB'] = 12821.6
        hydrogen['PaG'] = 10941.1
        return hydrogen[line]

    def Alcala_scaling(self, line, A=None, B=None, wave=None):
        '''
        extracts polynomials from Lacc-Lline scaling relationships from Alcala+2017 
        Parameters
        --------
        line : str
            emission line, must be in the form "Ha" or "PaB"
        A, B : floats (optional)
            user provided polynomials
        wave : float (optional)
            user provided emission line wavelength
        '''
        # open file
        alcala_lines = pd.read_csv('alcala2017_linear_fits.csv', comment='#')
        # user provided information
        if line is None:
            a = A
            b = B
            wave=None
            if a is None:
                raise ValueError('must provide A and B')
        # extract lines
        elif line in alcala_lines.Diagnostic.values: 
            a = alcala_lines['a'].loc[alcala_lines['Diagnostic']==line].values[0]
            b = alcala_lines['b'].loc[alcala_lines['Diagnostic']==line].values[0]
            aerr = alcala_lines['a_err'].loc[alcala_lines['Diagnostic']==line].values[0]
            berr = alcala_lines['b_err'].loc[alcala_lines['Diagnostic']==line].values[0]
            wave = alcala_lines['wavelength'].loc[alcala_lines['Diagnostic']==line].values[0]
        else:            
            raise ValueError(f'Line not found: please use either None or one of the following lines: {alcala_lines.Diagnostic.to_list()}')
        return a, aerr, b, berr


    def measure_lineflux(self, line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs):
        '''
        quick function to just return line flux
        
        '''
        return self.measure_EW_lineflux(line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, 
        only_line_flux=True, **kwargs)

    def measure_EW(self, line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs):
        '''
        quick function to just return EW
        
        '''
        return self.measure_EW_lineflux(line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, 
            only_EW=True, **kwargs)

    def measure_EW_lineflux_3sigma(self, line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs):
        '''
        calculate upper limit line emission and EW for non-detections.  USES SOAR/TSPEC4.1 SPECIFIC DELTA LAMBDA DISPERSION! MUST BE UPDATED FOR DIFFERENT TELESCOPE/INSTRUMENT!! 
        
        Parameters
        -------
        line : str
            emission line to calculate upper limit
        center_wavelength : float (optional)
            wavelength at center of emission line if different from the lab derived value
        xmin, xmax : float (optional)
            min and max wavelength to measure continuum around line
        exclude_min, exclude_max : float (optional)
            inner and outer wavelength around emission line to include
            
        Returns
        -------
        EW_upp : float 
            3σ upper limit for EW
        LineFlux_upp : float
            3σ upper limit for line flux
        
        '''
        
        real_center_wavelength = self.hydrogen_lines(line)
        if xmin is None:
            xmin = real_center_wavelength-35
        if xmax is None:
            xmax = real_center_wavelength+35
        # just get region of interest
        region = SpectralRegion(xmin*u.AA, xmax*u.AA)
        sub_spectrum = extract_region(self.spec, region)
    
        # find peak "center" of line
        if center_wavelength is None:
            center_wavelength = real_center_wavelength

        fitted_continuum = fit_continuum(sub_spectrum)
        y_fit = fitted_continuum(sub_spectrum.spectral_axis)

        # normalized spectrum for line flux measurement
        spec_normalized = sub_spectrum-y_fit
        spec_normalizedEW = sub_spectrum/y_fit

        region_cont = SpectralRegion([((xmin)*u.AA, (center_wavelength-5)*u.AA), ((center_wavelength+5)*u.AA, (xmax)*u.AA)])
        SS2 = extract_region(spec_normalized, region)
        SS2flux = SS2.flux.value 
        local_cont_mean = np.nanmean(SS2flux)
        local_cont_std = np.nanstd(SS2flux)
        
        region_line = SpectralRegion((real_center_wavelength-3.5)*u.AA, (real_center_wavelength+3.5)*u.AA)
        len_line = extract_region(spec_normalized, region_line)
        
        # SPECIFIC FOR SOAR!!!!  A/PIXEL measurement
        if line == 'BrG':
            deltalambda = 2.877
        elif line == 'PaB':
            deltalambda=1.72
        else:
            deltalambda=1.43
        LineFlux_upp = 3 * local_cont_std * deltalambda * np.sqrt(len(len_line.flux))
        
        Equivalent_Width = equivalent_width(spec_normalizedEW, regions=SpectralRegion((real_center_wavelength-3.5)*u.AA, (real_center_wavelength+3.5)*u.AA), continuum=1)
        SNR = local_cont_mean / local_cont_std
        deltalambda2 = 7 # A
        EW_upp = 3 * np.sqrt(2) * (deltalambda2 + Equivalent_Width.value ) / SNR
        
        IND = np.where((spec_normalizedEW.spectral_axis.value >= real_center_wavelength-3.5) & 
        (spec_normalizedEW.spectral_axis.value <= real_center_wavelength+3.5))
        avg_flux = np.nanstd(spec_normalizedEW.flux.value[IND])
        EW_upp = 3 * avg_flux * deltalambda2
        
        return EW_upp, LineFlux_upp

     
    
    def measure_EW_lineflux(self, line, center_wavelength=None, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs):
        '''
        calculate line emission and EW for given line.  MAIN WORKHORSE OF CODE!   
        *** USES SOAR/TSPEC4.1 SPECIFIC DELTA LAMBDA DISPERSION! MUST BE UPDATED FOR DIFFERENT TELESCOPE/INSTRUMENT!! ***
        
        Parameters
        -------
        line : str
            emission line to calculate upper limit
        center_wavelength : float (optional)
            wavelength at center of emission line if different from the lab derived value
        xmin, xmax : float (optional)
            min and max wavelength to measure continuum around line
        exclude_min, exclude_max : float (optional)
            inner and outer wavelength around emission line to include
        **kwargs:
            flat : boolean 
                whether or not to use flat continuum
            interactive : boolean
                interactive form to select detection or nondetection for line
            xmincon, xmaxcon : float
                decrease xmin and xmax by specific amount in A for continuum calculation
            innercon : float
                change inner exclusion region around emission line for continuum calculation in A
            plot : boolean
                plot final fit or not
            only_EW : boolean
                only return EW values
            only_line_flux : boolean
                only return line flux values
            
        Returns
        -------
        [LFmu, LFsigma] : list
            Line flux and uncertainty in units of erg/s/cm2 
        [EWmu, EWsigma] : list
            EW and uncertainty in A
        '''
        real_center_wavelength = self.hydrogen_lines(line)
        if xmin is None:
            xmin = real_center_wavelength-50
        if xmax is None:
            xmax = real_center_wavelength+50

        # just get region of interest
        region = SpectralRegion(xmin*u.AA, xmax*u.AA)
        sub_spectrum = extract_region(self.spec, region)
        # find peak "center" of line
        if center_wavelength is None:
            center_wavelength = sub_spectrum.spectral_axis.value[np.argmax(sub_spectrum.flux.value)]
        if abs(center_wavelength - real_center_wavelength) > 15:
            center_wavelength = real_center_wavelength
        # take region around line and find continuum
        region_cont = [(xmin*u.AA, (center_wavelength-10)*u.AA), ((center_wavelength+10)*u.AA, xmax*u.AA)]
        
        if kwargs.get('flat'):

            fitted_continuum = fit_continuum(sub_spectrum, window=region_cont)
            y_fit_cont = fitted_continuum(sub_spectrum.spectral_axis)

            #continuum
            y_fit = np.ones_like(y_fit_cont.value) * np.nanmean(y_fit_cont)
        else:
            fitted_continuum = fit_continuum(sub_spectrum)
            y_fit = fitted_continuum(sub_spectrum.spectral_axis)

        # normalized spectrum for line flux measurement
        spec_normalized = sub_spectrum - y_fit
        
        xmincon, xmaxcon = kwargs.get('xmincon', 0), kwargs.get('xmaxcon',0)
        innercon = kwargs.get('innercon',5)
        
        print('peak: ', np.nanmax(spec_normalized.flux.value))
        region_cont = SpectralRegion( [((xmin+xmincon)*u.AA, (center_wavelength-innercon)*u.AA),
                                       ((center_wavelength+innercon)*u.AA, (xmax-xmaxcon)*u.AA) ])
                                     
                                     
        SS = extract_region(spec_normalized, region)
        SS2 = extract_region(spec_normalized, region_cont)
        if line == 'BrG':
            SS2flux = SS2[1].flux.value
        else:
            SS2flux = np.append(SS2[0].flux.value, SS2[1].flux.value)
        local_cont_mean = np.nanmean(SS2flux)
        local_cont_std = np.nanstd(SS2flux)
        print('cont std: ', np.nanstd(SS2flux))
        print('S/N: ', np.nanmax(SS.flux.value) /np.nanstd(SS2flux) )
        SS2EW = extract_region(sub_spectrum, region_cont)
        if line == 'BrG':
            SS2fluxEW = SS2EW[1].flux.value
        else:
            SS2fluxEW = np.append(SS2EW[0].flux.value, SS2EW[1].flux.value)
        local_cont_meanEW = np.nanmean(SS2fluxEW)
        local_cont_stdEW = np.nanstd(SS2fluxEW)
        
        # determine type of object (detection or upper limit) from plot.  Then use the input to determine how to proceed. 
        # ENTER = detection
        # 3 = 3sigma upper limit
        move_forward = 'go'
        if kwargs.get('interactive'):
            plt.figure(figsize=(3,2))
            plt.step(spec_normalized.spectral_axis, spec_normalized.flux+y_fit, color='k')
            plt.plot(sub_spectrum.spectral_axis, y_fit, color='r')
            plt.show()
            
            PASS = input('continue?')
            if PASS == '':
                move_forward = 'go'
            elif PASS == '3':
                move_forward = 'upper'      
            else:
                move_forward = 'pass'
                
        # if detection
        if move_forward == 'go':
            # fit gaussian to line
            newx= np.linspace(spec_normalized.spectral_axis.min(), spec_normalized.spectral_axis.max(),200)
            Gaussian_model = GaussianModel()
            params = Gaussian_model.guess(spec_normalized.flux.value, x=spec_normalized.spectral_axis.value)
            params['sigma'].set(value=1.5, vary=True, expr='', min=0, max=7) # force gamma to be > 0
            params['center'].set(value=real_center_wavelength, vary=True, expr='', min=real_center_wavelength-15, max=real_center_wavelength+15) # force gamma to be > 0
            
            # try to fit gaussian to line
            try:
                data_bestfit  = Gaussian_model.fit(spec_normalized.flux.value, params, x=spec_normalized.spectral_axis.value)
                
                vals = data_bestfit.params
                ynew = Gaussian_model.eval(data_bestfit.params, x=newx.value) * spec_normalized.unit
                # get uncertainty of smooth model
                dely = data_bestfit.eval_uncertainty(x=newx.value) * spec_normalized.unit
                # make Gaussian profile spectrum
                un = StdDevUncertainty(dely)
                
                f1 = interpolate.interp1d(sub_spectrum.spectral_axis, y_fit)
                y_fit_interp = f1(newx) * spec_normalized.unit
                Gaussian_spectrum = Spectrum1D(spectral_axis = newx, flux=ynew+y_fit_interp, uncertainty=un)
                
                # measure Line Flux
                Line_flux = line_flux(Gaussian_spectrum-y_fit_interp)
                Line_flux = Line_flux.to(u.erg/u.cm**2./u.s)
                
                if exclude_min is None:
                    exclude_min = vals['center'].value - (3*vals['sigma'].value)
                if exclude_max is None:
                    exclude_max = vals['center'].value + (3*vals['sigma'].value)
                
                region_line = SpectralRegion(exclude_min*u.AA, exclude_max*u.AA)
                len_line = extract_region(spec_normalized, region_line)
                
                # dispersion
                if line == 'BrG':
                    deltalambda = 2.877
                elif line == 'PaB':
                    deltalambda=1.72
                else:
                    deltalambda=1.43
                    
                Line_fluxerr = local_cont_std * deltalambda * np.sqrt(len(len_line.flux))
            
                # measure EW
                # normalized spectrum for EW
                spec_normalizedEW = (Gaussian_spectrum / y_fit_interp)
                
                Equivalent_Width = equivalent_width(spec_normalizedEW, regions=SpectralRegion(exclude_min*u.AA, exclude_max*u.AA), continuum=1)
                
                SNR = local_cont_meanEW / local_cont_stdEW
                deltalambda2 = exclude_max - exclude_min
                EWerr_avg = np.sqrt(1+(local_cont_meanEW/Line_flux.value)) * ((deltalambda2 + Equivalent_Width.value) / SNR)
   
                # final results
                LFmu, LFsigma = Line_flux.value, Line_fluxerr
                LFresult = [Line_flux.value, Line_fluxerr]
                EWmu, EWsigma=Equivalent_Width.value, EWerr_avg
                EWresult = [Equivalent_Width.value, EWerr_avg]

                if kwargs.get('plot'):
                    fig, ax1 = plt.subplots(figsize=(5.5, 3), nrows=1, ncols=1, dpi=150)
                    ax = {'spectra':ax1}
                    # Plot the original spectrum and the fitted.
                    ax['spectra'].step(spec_normalized.spectral_axis, spec_normalized.flux, color='k')
                    ax['spectra'].plot(newx, ynew, color='blue')
                    ax['spectra'].fill_between(newx.value, ynew.value-dely.value, ynew.value+dely.value, color='lightblue')
                    ax['spectra'].axvline(exclude_min, color='gray', linestyle=':')
                    ax['spectra'].axvline(exclude_max, color='gray', linestyle=':')
                    ax['spectra'].axhline(0, color='orange')
                    ax['spectra'].text(0.02, 0.8, 
                        'EW = ' + format(EWmu,'.3f') + r' $\pm$ ' + format(EWsigma,'.3f') + ' $\AA$' + '\n$f$ = ' + format(LFmu,'.3e') + r' $\pm$ ' + format(LFsigma,'.3e') + ' erg/s/cm$^2$', 
                        transform=ax['spectra'].transAxes, fontsize=8,
                        bbox=dict(facecolor='lightgray', ec='none', pad=0.3, boxstyle='round'))

                    ax['spectra'].set_ylim(top=np.max(ynew.value+dely.value) * 1.5)
                    ax['spectra'].set_xlabel('wavelength ($\AA$)')
                    ax['spectra'].set_ylabel('Flux (erg/s/cm$^2/\AA$)')
                    plt.show()

                if kwargs.get('only_EW'):
                    print('EW: ', EWmu, '+/- ', EWsigma, ' A')
                    return [EWmu*u.AA, EWsigma*u.AA] 
                elif kwargs.get('only_line_flux'):
                    print('Fline: ', LFmu, '+/- ', LFsigma, ' erg/s/cm^2')
                    return [LFmu*u.erg / u.s/ u.cm**2, LFsigma*u.erg / u.s/ u.cm**2]  
                else:
                    print('EW: ', EWmu, '+/- ', EWsigma, ' A')
                    print('Fline: ', LFmu, '+/- ', LFsigma, ' erg/s/cm^2')
                    return [LFmu*u.erg / u.s/ u.cm**2, LFsigma*u.erg / u.s/ u.cm**2], [EWmu*u.AA, EWsigma*u.AA] 

            # if fitting gaussian fails calculate upper limit
            except:
                print('No spectral line')
                print('calculating upper limit')
                EW_upp, LineFlux_upp = self.measure_EW_lineflux_3sigma(line, center_wavelength=real_center_wavelength, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs)
                if kwargs.get('plot'):
                    fig, ax1 = plt.subplots(figsize=(5.5, 3), nrows=1, ncols=1, dpi=150)
                    ax = {'spectra':ax1}
                    # Plot the original spectrum and the fitted.
                    ax['spectra'].step(spec_normalized.spectral_axis, spec_normalized.flux, color='k')
                    ax['spectra'].axvline(real_center_wavelength-2, color='gray', linestyle=':')
                    ax['spectra'].axvline(real_center_wavelength+2, color='gray', linestyle=':')
                    ax['spectra'].axhline(0, color='orange')
                    ax['spectra'].text(0.02, 0.9, 
                        '$f$ < ' + format(LineFlux_upp.value,'.3e') + ' erg/s/cm$^2$', 
                        transform=ax['spectra'].transAxes, fontsize=8,
                        bbox=dict(facecolor='lightgray', ec='none', pad=0.3, boxstyle='round'))
                    plt.show()
                print('Fline: <', LineFlux_upp )
                print('EW: <', EW_upp )
                return [LineFlux_upp* u.erg / u.s/ u.cm**2, np.nan* u.erg / u.s/ u.cm**2], [EW_upp*u.AA, np.nan*u.AA]
        # calculate upper limit
        elif move_forward == 'upper':
            print('calculating upper limit')
            EW_upp, LineFlux_upp = self.measure_EW_lineflux_3sigma(line, center_wavelength=real_center_wavelength, xmin=None, xmax=None, exclude_min=None, exclude_max=None, **kwargs)
            if kwargs.get('plot'):
                fig, ax1 = plt.subplots(figsize=(5.5, 3), nrows=1, ncols=1, dpi=150)
                ax = {'spectra':ax1}
                # Plot the original spectrum and the fitted.
                ax['spectra'].step(spec_normalized.spectral_axis, spec_normalized.flux, color='k')
                ax['spectra'].axvline(real_center_wavelength-2, color='gray', linestyle=':')
                ax['spectra'].axvline(real_center_wavelength+2, color='gray', linestyle=':')
                ax['spectra'].axhline(0, color='orange')
                ax['spectra'].text(0.02, 0.9, 
                    '$f$ < ' + format(LineFlux_upp,'.3e') + ' erg/s/cm$^2$', 
                    transform=ax['spectra'].transAxes, fontsize=8,
                    bbox=dict(facecolor='lightgray', ec='none', pad=0.3, boxstyle='round'))
                plt.show()
            print('Fline: <', LineFlux_upp* u.erg / u.s/ u.cm**2 )
            print('EW: <', EW_upp*u.AA )
            return [LineFlux_upp* u.erg / u.s/ u.cm**2, np.nan* u.erg / u.s/ u.cm**2], [EW_upp*u.AA, np.nan*u.AA]
        else:
            return 'passing...', None
        
        
    def measure_line_luminosity(self, Line_Flux, Distance):
        '''
        measure line luminosity 
        
        Parameters
        -------
        Line_Flux : list
            list of [line flux, line flux err] in erg/s/cm2
        Distance : list
            list of [distance, distance err] in pc
            
        Returns
        ------
        [Line_Luminosity_Lsun, Line_Luminosity_Lsun_err] : list
            list of line luminosity and err in Lsun
        
        '''
        line_flux, line_flux_err = Line_Flux 
        distance, distance_err = Distance
        # line luminosity
        if isinstance(line_flux, float):
            line_flux = line_flux * u.erg / u.s/ u.cm**2
        if isinstance(line_flux_err, float):
            line_flux_err = line_flux_err * u.erg / u.s/ u.cm**2

        dist = distance * u.pc
        distance_err = distance_err * u.pc
        Line_Luminosity = 4 * np.pi * dist**2. * line_flux
        Line_Luminosity_err = Line_Luminosity * np.sqrt((2*distance_err/dist)**2. + (line_flux_err/line_flux)**2.)

        # convert to Lsun
        Line_Luminosity_Lsun = Line_Luminosity.to(u.Lsun)
        Line_Luminosity_Lsun_err = Line_Luminosity_err.to(u.Lsun)
        if not np.isnan(Line_Luminosity_Lsun_err.value):
            print('Lline_Lsun: ', Line_Luminosity_Lsun, '+/-', Line_Luminosity_Lsun_err)
        else: 
            print('Lline_Lsun: <', Line_Luminosity_Lsun)
        return [Line_Luminosity_Lsun, Line_Luminosity_Lsun_err]

    def measure_accretion_luminosity(self, line, Line_Luminosity):
        '''
        measure accretion luminosity 
        
        Parameters
        -------
        Line_Luminosity : list
            list of [line luminosity, line luminosity err] in Lsun
        line : str
            emission line that has been calculated
            
        Returns
        ------
        [Lacc_Lsun, Lacc_Lsun_err] : list
            accretion luminosity and error in Lsun
        [np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit, logsigma_Lacc*Lacc_Lsun.unit] : list
            log of accretion luminosity and log error in Lsun
        
        '''
        line_lum, line_lum_err = Line_Luminosity
        line_lum_err_LOG = 0.434 * line_lum_err/line_lum

        # scaling relation
        a,a_err,b,b_err = self.Alcala_scaling(line)
        Lacc_Lsun = 10**(a * np.log10(line_lum.value) + b)*u.Lsun

        #uncertainty propagation
        log10L = np.log10(line_lum.value)
        sigma_log10L = 0.434 * line_lum_err.value/line_lum.value

        alog10L = a * log10L
        alog10Lb = a * log10L + b 

        logsigma_Lacc = np.sqrt( b_err**2. + (alog10L * np.sqrt( (a_err/a)**2. + (sigma_log10L/log10L)**2. ))**2.) 

        sigma_Lacc = 2.303 * (10**(alog10Lb)) * logsigma_Lacc
        Lacc_Lsun_err = sigma_Lacc * u.Lsun

        if not np.isnan(Lacc_Lsun_err.value):
            print('LOG Lacc_Lsun: ', np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit, '+/-', logsigma_Lacc*Lacc_Lsun.unit)
            print('Lacc_Lsun: ', Lacc_Lsun, '+/-', Lacc_Lsun_err)
        else:
            print('LOG Lacc_Lsun: <', np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit)
            print('Lacc_Lsun: <', Lacc_Lsun)
        return [Lacc_Lsun, Lacc_Lsun_err], [np.log10(Lacc_Lsun.value)*Lacc_Lsun.unit, logsigma_Lacc*Lacc_Lsun.unit]

    def measure_Mdot(self, mass, radius, acc_lum, Rin=5*u.Rsun):
        '''
        measure accretion rate 
        
        Parameters
        -------
        mass : list
            list of [mass, mass err] in Msun
        radius : list
            list of [radius, radius err] in Rsun
        acc_lum : list
            list of [accretion luminosity, accretion luminosity err] in Lsun
        Rin : Quantity (optional)
            truncation radius for Mdot calculation
            
        Returns
        ------
        [LOGmass, LOGmass_err] : list
            list of log mass and log mass err in Msun
        [LOGMdot, LOGMdot_err] : list
            list of log Mdot and log Mdot err in Mjup/yr
        
        '''
        Lacc, Lacc_err = acc_lum
        # convert to Mdot
        Radius_Rsun = radius[0] * u.Rsun
        Radius_Rsun_err = radius[1] * u.Rsun
        Mass_Msun = mass[0] * u.Msun
        Mass_Msun_err = mass[1] * u.Msun
        G = const.G.cgs
        
        Mdot = ((1-(Radius_Rsun/Rin))**-1) * (Radius_Rsun*Lacc)/(const.G*Mass_Msun)

        Mdot_err = Mdot * np.sqrt((Lacc_err/Lacc)**2. + (Radius_Rsun_err/Radius_Rsun)**2. + (Mass_Msun_err/Mass_Msun)**2.)

        LOGmass = np.log10(Mass_Msun.value)*u.Msun
        LOGmass_err = 0.434 * Mass_Msun_err.value/Mass_Msun.value  *u.Msun

        LOGMdot = np.log10(Mdot.to(u.Mjup/u.yr).value)*u.Mjup/u.yr
        LOGMdot_err = 0.434 * Mdot_err.to(u.Mjup/u.yr) / Mdot.to(u.Mjup/u.yr)
        if not np.isnan(Mdot_err.value):
            print('Mdot Mj/yr:', Mdot.to(u.Mjup/u.yr), '+/-',Mdot_err.to(u.Mjup/u.yr))
            print()
            print('Log(Mdot): ', LOGMdot , '+/-',LOGMdot_err )
        else:
            print('Mdot: <', Mdot.to(u.Mjup/u.yr))
            print()
            print('Log(Mdot): <', LOGMdot)
        print('Log(Mass):', LOGmass, '+/-', LOGmass_err)

        return [LOGmass, LOGmass_err],[ LOGMdot, LOGMdot_err]

    def run_measure_object(self, line, Distance, mass, radius, center_wavelength=None, xmin=None, xmax=None, 
    exclude_min=None, exclude_max=None, Rin=5*u.Rsun,
    plot=True, interactive=False, **kwargs):
        '''
        MAIN WORKHORSE to run all functions and calculate line emission -> Mdot
        
        Parameters
        -------
        line : float
            emission line to measure
        Distance : list
            list of [distance, distance err] in pc
        mass : list
            list of [mass, mass err] in Msun
        radius : list
            list of [radius, radius err] in Rsun
        center_wavelength : float (optional)
            wavelength at center of emission line if different from the lab derived value
        xmin, xmax : float (optional)
            min and max wavelength to measure continuum around line
        exclude_min, exclude_max : float (optional)
            inner and outer wavelength around emission line to include
        Rin : Quantity (optional)
            truncation radius for Mdot calculation
        plot : boolean
                plot final fit or not
        interactive : boolean
                interactive form to select detection or nondetection for line
                
        **kwargs
            flat : boolean 
                whether or not to use flat continuum
            xmincon, xmaxcon : float
                decrease xmin and xmax by specific amount in A for continuum calculation
            innercon : float
                change inner exclusion region around emission line for continuum calculation in A
            only_EW : boolean
                only return EW values
            only_line_flux : boolean
                only return line flux values 
        '''
        
        # measure EW and line flux
        LFresult, EWresult = self.measure_EW_lineflux(line, center_wavelength=center_wavelength, xmin=xmin, 
            xmax=xmax, exclude_min=exclude_min, exclude_max=exclude_max, plot=plot, interactive=interactive, **kwargs)
        
        if not isinstance(LFresult, str):
            # measure line luminosity
            Line_Luminosity = self.measure_line_luminosity(LFresult, Distance)
            # measure accretion luminosity
            Lacc, logLacc = self.measure_accretion_luminosity(line, Line_Luminosity)
            # measure accretion rate
            mass, Mdot = self.measure_Mdot(mass, radius, Lacc, Rin=Rin)
            # print values
            if EWresult is not None:
                print()
                print(EWresult[0].value, EWresult[1].value, LFresult[0].value, LFresult[1].value, logLacc[0].value,
                     logLacc[1].value, Lacc[1].value, Mdot[0].value, Mdot[1].value)
            else:
                print()
                print('','', LFresult[0].value, '', logLacc[0].value,
                     '', '', Mdot[0].value)
        
