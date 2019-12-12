# pyFADO: A tool to extract info from FADO result files

FADO: http://spectralsynthesis.org/fado.html

## Basic description 

The following ``classes`` are implemented

* ``FadoLoad``: Load the files and stores the most important information
 
* ``OneD`` : Manages the results that are stored as spectra

* ``EmLines`` : Manages the results for the emission lines

Core functionality implemented

## Requirements

    numpy
    astropy
    specutils
    python-box
    matplotlib (for the old ``FADO`` class)
    
## Installation

    git clone https://github.com/miguelverdugo/pyFADO.git

or

    wget https://github.com/miguelverdugo/pyFADO/archive/master.zip
    
and then
    
    pip install -e .
    
or

    python setup.py install --path <path>
    







