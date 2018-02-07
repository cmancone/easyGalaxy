#!/usr/bin/python

import sys, os, pyfits, math
import numpy as np
import ezgal

if len(sys.argv) < 3:
    print '\nconvert pegase2 sps models to EzGal fits format'
    print 'Usage: convert_pegase2.py pegase.fits output.model\n'
    sys.exit(2)

filein = sys.argv[1]
fileout = sys.argv[2]

if not os.path.isfile(filein):
    raise ValueError('Input file does not exist or is not readable!')

# try to extract meta data out of fileout
sfh = 'ssp'
met = ''
imf = ''
# split on _ but get rid of the extension
parts = '.'.join(fileout.split('.')[:-1]).split('_')
# metallicity
if parts.count('z'):
    met = parts[parts.index('z') + 1]
# imf
for (check, val) in zip(['krou', 'salp', 'chab'],
                        ['Kroupa', 'Salpeter', 'Chabrier']):
    if parts.count(check):
        imf = val
        break

# read in input file
fits = pyfits.open(filein)

# read in wavelengths
ls = fits['ets_cont_wca'].data['bfit']
nls = ls.size

# speed of light in angstroms/s
c = ezgal.utils.convert_length(ezgal.utils.c, incoming='m', outgoing='a')

# read in seds
seds = fits[0].data

# conversion from Lo/Anstrom to ergs/s/Hz
seds *= 3.826e33 * (ls**2.0) / c

# conversion from ergs/s/Hz to ergs/s/Hz/cm^2.0
seds /= (4.0 * math.pi * ezgal.utils.convert_length(
    10, incoming='pc', outgoing='cm')**2.0)

# convert from angstroms to hertz
vs = c / ls

# sort in frequency space
sinds = vs.argsort()

# transpose sed array
seds = seds.transpose()

# generate fits frame with sed in it
primary_hdu = pyfits.PrimaryHDU(seds[sinds, :])
primary_hdu.header.update('units', 'ergs/s/cm^2/Hz')
primary_hdu.header.update('has_seds', True)
primary_hdu.header.update('nfilters', 0)
primary_hdu.header.update('nzfs', 0)

# store meta data
if sfh and met and imf:
    primary_hdu.header.update('has_meta', True)
    primary_hdu.header.update('model', 'p2', comment='meta data')
    primary_hdu.header.update('met', met, comment='meta data')
    primary_hdu.header.update('imf', imf, comment='meta data')
    primary_hdu.header.update('sfh', sfh, comment='meta data')

# fetch fits extension which contains age/mass info
info = fits['ETS_PARA'].data

# store the list of frequencies in a table
vs_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(
    name='vs', array=vs[sinds],
    format='D', unit='hertz')]))
# and the list of ages + masses
ages_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(
    name='ages', array=info['age'] * 1e6,
    format='D', unit='years'), pyfits.Column(name='masses',
                                             array=info['mstars'] + info['mwd']
                                             + info['mbhns'],
                                             format='D',
                                             unit='m_sun')]))
ages_hdu.header.update('has_mass', True)

# make the fits file in memory
hdulist = pyfits.HDUList([primary_hdu, vs_hdu, ages_hdu])
# and write it out
hdulist.writeto(fileout, clobber=True)
