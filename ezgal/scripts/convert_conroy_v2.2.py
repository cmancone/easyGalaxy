#!/usr/bin/python

import sys, os, pyfits, math
import numpy as np
from ezgal import utils

if len(sys.argv) < 4:
    print '\nconvert conroy sps models to ez_galaxy fits format'
    print 'Usage: convert_conroy.py csp_model.out.spec model.lamba csp_model.out\n'
    sys.exit(2)

filein = sys.argv[1]
lambdas = sys.argv[2]
fileout = sys.argv[3]

if not os.path.isfile(filein):
    raise ValueError('Input file does not exist or is not readable!')
if not os.path.isfile(lambdas):
    raise ValueError('Input file does not exist or is not readable!')

# try to extract meta data out of fileout
sfh = ''
tau = ''
met = ''
imf = ''
# split on _ but get rid of the extension
parts = '.'.join(fileout.split('.')[:-1]).split('_')
# look for sfh
for (check, val) in zip(['ssp', 'exp'], ['SSP', 'Exponential']):
    if parts.count(check):
        sfh = val
        sfh_index = parts.index(check)
        break
# tau?
if sfh:
    tau = parts[sfh_index + 1] if sfh == 'Exponential' else ''
# metallicity
if parts.count('z'):
    met = parts[parts.index('z') + 1]
# imf
for (check, val) in zip(['krou', 'salp', 'chab'],
                        ['Kroupa', 'Salpeter', 'Chabrier']):
    if parts.count(check):
        imf = val
        break

# read in wavelengths
ls = []
fp = open(lambdas, 'r')
ls = np.array([line.strip() for line in fp]).astype('float')
fp.close()
nls = ls.size

# conversion from Lo/Hz to ergs/s/Hz/cm^2.0
conv = 3.826e33 / (4.0 * math.pi * utils.convert_length(
    10, incoming='pc', outgoing='cm')**2.0)
vs = utils.convert_length(utils.c, incoming='m', outgoing='a') / ls

# now read in the model file
fp = open(filein, 'r')
ages = []
masses = []
c = 0
while True:

    # each age takes up two lines
    descr = fp.readline()
    if not descr: break
    if descr[0] == '#': continue

    # skip the first data line (it just gives the number of ages)
    if not c:
        c += 1
        continue

    # read age and mass and convert to years
    parts = descr.split()
    ages.append(10.0**float(parts[0].strip()))
    masses.append(10.0**float(parts[1].strip()))

    # now get sed data
    data = np.array(fp.readline().strip().split()).reshape(
        (nls, 1)).astype('float')
    # and convert from Lo/Hz to ergs/s/Hz
    data *= conv

    if c == 1:
        seds = data
    else:
        seds = np.hstack((seds, data))

    c += 1
ages = np.array(ages)

# sort in frequency space
sinds = vs.argsort()

# generate fits frame with sed in it
primary_hdu = pyfits.PrimaryHDU(seds[sinds, :])
primary_hdu.header.update('units', 'ergs/s/cm^2/Hz')
primary_hdu.header.update('has_seds', True)
primary_hdu.header.update('nfilters', 0)
primary_hdu.header.update('nzfs', 0)

# store meta data
if sfh and met and imf:
    primary_hdu.header.update('has_meta', True)
    primary_hdu.header.update('model', 'C09', comment='meta data')
    primary_hdu.header.update('met', met, comment='meta data')
    primary_hdu.header.update('imf', imf, comment='meta data')
    primary_hdu.header.update('sfh', sfh, comment='meta data')
    if sfh == 'Exponential':
        primary_hdu.header.update('tau', tau, comment='meta data')

# store the list of frequencies in a table
vs_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(
    name='vs', array=vs[sinds],
    format='D', unit='hertz')]))
# and the list of ages + masses
ages_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(
    name='ages', array=ages, format='D',
    unit='years'), pyfits.Column(
        name='masses', array=masses,
        format='D', unit='m_sun')]))
ages_hdu.header.update('has_mass', True)

# make the fits file in memory
hdulist = pyfits.HDUList([primary_hdu, vs_hdu, ages_hdu])
# and write it out
hdulist.writeto(fileout, clobber=True)
