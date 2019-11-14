import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.constants import c 

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

        values = data[:,xmin:xmax][0]
        
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
        
        
        
        
        
        
        
