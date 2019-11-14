#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:57:48 2019

@author: mverdugo
"""

from pyFADO import FADO
from astropy.io import fits


fado  = FADO("spec_wcs1d_p17q10_new_der.fits.output3_EL.fits",
             "spec_wcs1d_p17q10_new_der.fits.output3_ST.fits",
             "spec_wcs1d_p17q10_new_der.fits.output3_1D.fits",
             "spec_wcs1d_p17q10_new_der.fits.output3_DE.fits")

fado.files()

#fado.open_the_files()

ELs = fado.ELnames()
print(ELs)
print(fado.waves())

rows = fado.ELrows()
print(rows)

print(fado.get_results("Hbeta"))

fado.plot_fit_to_line("[OIII]5006",window=100, color='r')
fado.plot_line_residual("[OIII]5006",window=100, color='b')


#fado.plot_line_residual('Halpha', window=200)