# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.constants import c
import astropy.wcs as fitswcs
import astropy.units as u
from astropy.modeling import models
from astropy.table import Table

from box import Box
from specutils import Spectrum1D




class FadoLoad:
    """
    This class is meant to load the necessary files and keep them in memory
    to be used in subsequent classes.

    It should also read some important information
    """
    def __init__(self, EL_file, ST_file, ONED_file, DE_file):

        self.EL_file = EL_file
        self.ST_file = ST_file
        self.ONED_file = ONED_file
        self.DE_file = DE_file
        self.wave_unit = u.AA

    @property
    def EL(self):
        with fits.open(self.EL_file, lazy_load_hdus=False) as hdu:
            return Box({"header": hdu[0].header, "data": hdu[0].data})

    @property
    def ST(self):
        with fits.open(self.ST_file, lazy_load_hdus=False) as hdu:
            return Box({"header": hdu[0].header, "data": hdu[0].data})

    @property
    def ONED(self):
        with fits.open(self.ONED_file, lazy_load_hdus=False, memmap=True) as hdu:
            return Box({"header": hdu[0].header, "data": hdu[0].data})

    @property
    def DE(self):
        with fits.open(self.DE_file, lazy_load_hdus=False) as hdu:
            return Box({"header": hdu[0].header, "data": hdu[0].data})

    @property
    def redshift(self):
        try:
            z = self.ONED.header["REDSHIFT"]
        except KeyError as e:
            print(e)
            print("Redshift key not found in", self.ONED_file)
            raise
        else:
            return z

    @property
    def fado_ver(self):
        try:
            ver = self.ONED.header["FADO_VER"]
        except KeyError as e:
            print(e)
            print("version key not found in", self.ONED_file)
            raise
        else:
            return ver

    @property
    def ext_law(self):
        try:
            law = self.ONED.header["R_LAWOPT"]
        except KeyError as e:
            print(e)
            print("version key not found in", self.ONED_file)
            raise
        else:
            return law

    @property
    def flux_unit(self):
        """

        Returns
        -------
        the scale to which the spectra must be multiplied to have correct flux levels
        """
        try:
            unit = u.Unit("erg / (s cm**2 Angstrom)" )
            norm = self.ONED.header["GALSNORM"]
            fluxunit = 10**self.ONED.header["FLUXUNIT"]
        except KeyError as e:
            print(e)
            print("keys not found in", self.ONED_file)
            raise
        else:
            return norm*fluxunit*unit


class FadoOneD:
    """
    Class to manage the results in the ONED File
    """
    def __init__(self, fado_load=FadoLoad):
        self.fado_load = fado_load
        self.header = fado_load.ONED.header
        self.data = fado_load.ONED.data
        self.max_rows = self.data.shape[0] + 1 # because python



    @property
    def wcs(self):
        wcs = fitswcs.WCS(header={'CDELT1': self.header["CDELT1"],
                        'CRVAL1': self.header["CRVAL1"],
                        'CUNIT1': 'Angstrom',
                        'CTYPE1': 'WAVE',
                        'CRPIX1': self.header["CRPIX1"]})
        return wcs

    def spectrum(self, row=1, row_name=None, scale=True):
        """
        Creates a specutils.Spectrum1D from FADO ONED files.
        the spectrum can be specified as a row or as a name.
        Allowed names are the following

        1: 'Observed'         spectrum de-redshifted and rebinned'
        2: 'Error'
        3: 'Mask'
        4: 'Best fit'
        5: 'Average'            of individual solutions
        6: 'Median'
        7: 'Stdev'
        8: 'Stellar'            using best fit'
        9: 'Nebular'            using best fit'
        10: 'AGN'               using best fit'
        11: 'M/L'
        12: 'LSF'  Line spread function

        The best-fits for individual solution can only be specified as a number 13-XX
        see self.max_rows

        Parameters
        ----------
        row: int, the row where the spectra is extracted, default=1 (0 in python notation)
        row_name: str, name of the row (optional)
        scale = bool, whether to scale the spectra or not, default True

        Returns
        -------
        a specutils.Spectrum1D object
        """
        row_names = {'observed': 1,  'error': 2,  'mask': 3,
                     'best fit': 4, 'average': 5,  'median': 6,
                     'stdev': 7,  'stellar': 8,  'nebular': 9,
                     'm/l': 10,   "lsf": 12}
        try:
            row = row_names[row_name.lower()]
        except (KeyError, AttributeError) as e:
            print("name not found", e)

        row = row - 1

        if scale is False:
            scale = 1
        else:
            scale = self.fado_load.flux_unit

        flux = self.data[row] * scale
        sp = Spectrum1D(flux=flux, wcs=self.wcs)
        return sp


class FadoEL:
    """
        Class to manage the results in the 1D files
    """

    def __init__(self, fado_load=FadoLoad):
        self.fado_load = fado_load
        self.scale = self.fado_load.flux_unit
        self.header = fado_load.EL.header
        self.data = fado_load.EL.data
        self.names = self.line_dicts["names"]
        self.waves = self.line_dicts["waves"]
        self.info = self.line_dicts["info"]


    @property
    def line_dicts(self):
        """ update the attributes of the lines fitted in the spectra
        self.line_names is a list with the names
        self.lines_waves = a dictionary with the rest-frame wavelengths
        self.line_info  = a dictionary with the pointers to the results
        """
        keys = list(self.header.keys())
        el_keys = [k for k in keys if "EL___" in k]
        el_names = []
        el_waves = {}
        el_info = {}
        for k in el_keys:
            [name, wave, rows] = self.header[k].split()

            name = name + "_" +wave.split(".")[0]
            el_names.append(name)
            el_waves.update({name: float(wave)})
            el_info.update({name: rows})

        return {'names': el_names, 'waves': el_waves, "info": el_info}

    def results(self, line_name, mode="best"):
        """
        returns a dictionary with the results of the fit for a emission line
        see self.names for names
        """

        rows = self.info
        window = rows[line_name].split('--')
        xmin = int(window[0])-1
        xmax = int(window[1])-1
        values = self.data[:, xmin:xmax][0]
        results = {'lambda': values[0] * u.AA,
                   'amplitude': values[1] * self.scale,
                   'sigma': values[2] * u.AA,
                   'vel': values[3] * u.km / u.s,
                   'shift': values[4] * u.AA,
                   'flux': values[5] * self.scale,
                   'ew': values[6] * u.AA}

        return results

    def errors(self, line_name, mode="best"):
        """
        returns a dictionary with the results of the fit for a emission line
        see ELnames for names
        """
        rows = self.info
        window = rows[line_name].split('--')
        xmin = int(window[0]) - 1
        xmax = int(window[1]) - 1
        values = self.data[:, xmin:xmax][1]
        errors = {'lambda': values[0] * u.AA,
                  'amplitude': values[1] * self.scale,
                  'sigma': values[2] * u.AA,
                  'vel': values[3] * u.km / u.s,
                  'shift': values[4] * u.AA,
                  'flux': values[5] * self.scale,
                  'ew': values[6] * u.AA}

        return errors

    def line_spectra(self, line_name):
        """
        Create a specutils.Spectrum1D for the line
        """
        results = self.results(line_name)
        lambda_r = results["lambda"]
        vel = results["vel"]
        sigma = results["sigma"]
        amplitude = results["amplitude"]

        center = (vel / c.to('km/s')) * lambda_r + lambda_r
        spectrum = FadoOneD(self.fado_load)
        wcs = spectrum.wcs

        length = self.fado_load.ONED.header["NAXIS1"]  # size of the spectrum in pixels
        sp = Spectrum1D(flux=np.zeros(length) * self.scale,
                        wcs=wcs)

        model = models.Gaussian1D(amplitude=amplitude,
                                  mean=center,
                                  stddev=sigma)
        sp = sp + Spectrum1D(spectral_axis=sp.spectral_axis, flux=model(sp.wavelength))
        return sp

    def to_table(self, mode="best"):
        """
        Create an astropy table with the results of the emission line fitting

        Parameters
        ----------
        mode: best, mean or median

        Returns
        -------
        astropy.Table
        """
        column_names = ("line_name",
                        "lambda",  "amplitude", "sigma", "vel", "shift", "flux", "ew",
                        "lambda_err",  "amplitude_err", "sigma_err", "vel_err", "shift_err", "flux_err", "ew_err")
        dtypes = ['S11'] + ['f8']*14
        table = Table(names=column_names, dtype=dtypes)

        for name in self.names:
            results = self.results(name, mode=mode)
            errors = self.errors(name, mode=mode)
            row = [name] + \
                  [results[key] for key in results.keys()] + \
                  [errors[key] for key in errors.keys()]
            table.add_row(row)

        return table



class FadoST:
    """
    Class to manage the results in the statistical file
    """

    pass


class FadoDE:
    """
    Class to manage the results in the DE files
    """
    pass


class FADO:
    """
    This class is supposed to load and extract parameters from FADO files

    """

    EL_file = None
    ST_file = None
    ONED_file = None
    DE_file = None

    def __init__(self, EL_file, ST_file, ONED_file, DE_file):

        self.EL_file = EL_file
        self.ST_file = ST_file
        self.ONED_file = ONED_file
        self.DE_file = DE_file

    # open_the_files
        self.hdu_EL = fits.open(self.EL_file)
        self.hdu_ST = fits.open(self.ST_file)
        self.hdu_ONED = fits.open(self.ONED_file)
        self.hdu_DE = fits.open(self.DE_file)

    def files(self):
        """
        just print the files being used
        """
        print(self.EL_file)
        print(self.ST_file)
        print(self.ONED_file)
        print(self.DE_file)

    def ELnames(self):
        """
        create dict of emission lines and their wavelengths
        """

        head = self.hdu_EL[0].header
        keys = list(head.keys())
        el_keys = [k for k in keys if "EL___" in k ]

        el_waves = {}
        for k in el_keys:
            [name, wave, rows] = head[k].split()
            if "[" in name:
                name = name + wave.split(".")[0]

            el_waves.update( {name: float(wave)})

        return el_waves

    def ELrows(self):
        """
        create dict of emission lines and their windows for data
        """

        head = self.hdu_EL[0].header
        keys = list(head.keys())
        el_keys = [k for k in keys if "EL___" in k ]

        el_rows = {}
        for k in el_keys:
            [name, wave, rows] = head[k].split()
            if "[" in name:
                name = name + wave.split(".")[0]

            el_rows.update( {name: rows} )

        return el_rows

    def waves(self):
        """
        Return the wavelength grid
        """
        head = self.hdu_ONED[0].header
        NAXIS1 = head["NAXIS1"]
        CRVAL1 = head["CRVAL1"]
        CRPIX1 = head["CRPIX1"] - 1 # because python
        CRDELT1 = head["CDELT1"]
        waves = CRVAL1 + CRDELT1*np.arange(CRPIX1, NAXIS1, 1)

        return waves

    def print_rows(self):
        """
        print what the rows of 1D files contain
        """
        head = self.hdu_ONED[0].header
        keys = list(head.keys())
        for k in keys:
            if "ROW" in k:
                print(k + '= ' ,repr(head[k]))

        print()
        print("Maximum row number=", head["NAXIS2"])

    def get_spectra(self, row):
        """
        return a numpy array with the spectra of that row
        see print_rows for description
        """
        data = self.hdu_ONED[0].data
        if row<1 and row>data.shape[1]:
            raise ValueError("row number not valid")

        spec = data[row-1]

        return spec

    def plot_spec(self, row, wmin=None, wmax=None, **kwargs):
        """
        plot a spectra from 1D files between wmin and wmax (default all)
        **kwargs are passed to matplotlib.pyplot
        """
        waves = self.waves()
        spec = self.get_spectra(row)
        if wmin is None:
            wmin = waves[0]
        if wmax is None:
            wmax = waves[-1]

        spec = spec[(waves>=wmin) & (waves<=wmax)]
        waves = waves[(waves>=wmin) & (waves<=wmax)]
        plt.plot(waves, spec, **kwargs)

    def plot_obs_spectrum(self, row=None, wmin=None, wmax=None, **kwargs):
        """
        plot the observed spectrum between wmin and wmax (default all)
        row number is optional in case observed spectrum is in another row
        check using print_rows
        """
        if row is None:
            row = 1

        self.plot_spec(row, wmin, wmax, **kwargs)

    def plot_model_spectrum(self, row=None, wmin=None, wmax=None, **kwargs):
        """
        plot the best fit FADO spec
        """
        if row is None:
            row = 4

        self.plot_spec(row, wmin, wmax, **kwargs)

    def plot_residual(self, wmin=None, wmax=None, **kwargs):
        """
        plot the residuals: observed - model
        """
        waves = self.waves()
        obs = self.get_spectra(1)
        model = self.get_spectra(4)
        if wmin is None:
            wmin = waves[0]
        if wmax is None:
            wmax = waves[-1]

        obs = obs[(waves>=wmin) & (waves<=wmax)]
        model = model[(waves>=wmin) & (waves<=wmax)]
        waves = waves[(waves>=wmin) & (waves<=wmax)]
        plt.plot(waves, obs - model, **kwargs)

    def plot_line(self, linename, window=None, **kwargs):
        """
        plot the region around a emission line
        for names see ELnames
        window is optional defaul 100A
        """
        if window is None:
            window = 100

        ELdict = self.ELnames()
        center = ELdict[linename]
        wmin = center - 0.5*window
        wmax = center + 0.5*window

        self.plot_obs_spectrum(row=1, wmin=wmin, wmax=wmax, **kwargs)

    def plot_line_residual(self, linename, window=None, **kwargs):
        """
        as above but after subtracting the model spectrum
        """
        if window is None:
            window = 100

        ELdict = self.ELnames()
        center = ELdict[linename]
        wmin = center - 0.5*window
        wmax = center + 0.5*window

        self.plot_residual(wmin=wmin, wmax=wmax, **kwargs)

    def get_results(self, linename):
        """
        returns a dictionary with the results of the fit for a emission line
        see ELnames for names
        """
        data = self.hdu_EL[0].data
        rows = self.ELrows()
        window = rows[linename].split('--')
        xmin = int(window[0])-1
        xmax = int(window[1])-1

        values = data[:, xmin:xmax][0]

        results = {'Lambda':values[0],
                   'Amplitude':values[1],
                   'Sigma':values[2],
                   'Vel':values[3],
                   'shift':values[4],
                   'Flux':values[5],
                   'EW':values[6]}

        return results

    def get_errors(self, linename):
        """
        returns a dictionary with the results of the fit for a emission line
        see ELnames for names
        """
        data = self.hdu_EL[0].data
        rows = self.ELrows()
        window = rows[linename].split('--')
        xmin = int(window[0])-1
        xmax = int(window[1])-1

        values = data[:,xmin:xmax][1]

        errors = {'Lambda':values[0],
                   'Amplitude':values[1],
                   'Sigma':values[2],
                   'Vel':values[3],
                   'shift':values[4],
                   'Flux':values[5],
                   'EW':values[6]}

        return errors

    def plot_fit_to_line(self, linename, window=None, **kwargs):
        """
        plot the fits of to a line
        works better with plot_line_residuals
        """
        results = self.get_results(linename)
        if window is None:
            window = 100

        ELdict = self.ELnames()
        center = ELdict[linename]
        wmin = center - 0.5*window
        wmax = center + 0.5*window

        lambda_r = results["Lambda"]
        vel = results["Vel"]
        mu = (vel/c.to('km/s').value) * lambda_r + lambda_r
        sigma = results["Sigma"]
        flux = results["Flux"]
        x = self.waves()
        x = x[(x>wmin) & (x<wmax)]
        y = flux*norm.pdf(x, mu, sigma)

        plt.plot(x, y, **kwargs)
