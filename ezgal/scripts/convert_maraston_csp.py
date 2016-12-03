#!/usr/bin/python

import sys,os,math,pyfits
import numpy as np
import utils
import scipy.interpolate as interpolate

if len( sys.argv ) < 3:
	print '\nconvert maraston csp models to ez_galaxy fits format'
	print 'Usage: convert_maraston_csp.py csp_model.in csp_model.out\n'
	sys.exit(2)

# was a file passed for the masses?
if len( sys.argv ) == 4:
	filein = sys.argv[1]
	filemass = sys.argv[2]
	fileout = sys.argv[3]
else:
	filein = sys.argv[1]
	fileout = sys.argv[2]
	filemass = False
if not os.path.isfile( filein ): raise ValueError( 'Input file does not exist or not readable!' )

# try to extract meta data out of fileout
sfh = ''; tau = ''; met = ''; imf = ''
# split on _ but get rid of the extension
parts = '.'.join( fileout.split( '.' )[:-1] ).split( '_' )
# look for sfh
for (check,val) in zip( ['ssp','exp'], ['SSP','Exponential'] ):
	if parts.count( check ):
		sfh = val
		sfh_index = parts.index( check )
		break
# tau?
if sfh:
	tau = parts[sfh_index+1] if sfh == 'Exponential' else ''
# metallicity
if parts.count( 'z' ):
	met = parts[ parts.index( 'z' ) + 1 ]
# imf
for (check,val) in zip( ['krou','salp','chab'], ['Kroupa', 'Salpeter', 'Chabrier'] ):
	if parts.count( check ):
		imf = val
		break

# speed of light
c = utils.convert_length( utils.c, incoming='m', outgoing='a' )

# read in model
fi = open( filein, 'r' )
res = []
for line in fi:
	if line[0] == '#': continue
	res.extend( line.strip().split() )
fi.close()

# convert to float and return original shape (m, 3)
arr = np.array( res ).reshape( (-1,3) ).astype( 'float' )
# grab unique ages
ages = np.unique( arr[:,0] )*1e9
# and unique ls
ls = np.unique( arr[:,1] )
# generate sed array
seds = arr[:,2].reshape( (ages.size,ls.size) ).transpose()
# convert from angstroms to hertz
vs = c/ls
# convert from ergs/s/A to ergs/s/Hz
seds *= ls.reshape( (ls.size,1) )**2.0/c
# and now from ergs/s/Hz to ergs/s/Hz/cm^2.0
seds /= (4.0*math.pi*utils.convert_length( 10, incoming='pc', outgoing='cm' )**2.0)

# sort in frequency space
sinds = vs.argsort()

# did the masses get passed?
if filemass:
	data = utils.rascii( filemass, silent=True )
	mass_interp = interpolate.interp1d( data[:,0]*1e9, data[:,1] )
	masses = mass_interp( ages )

# generate fits frame with sed in it
primary_hdu = pyfits.PrimaryHDU(seds[sinds,:])
primary_hdu.header.update( 'units', 'ergs/s/cm^2/Hz' )
primary_hdu.header.update( 'has_seds', True )
primary_hdu.header.update( 'nfilters', 0 )
primary_hdu.header.update( 'nzfs', 0 )

# store meta data
if sfh and met and imf:
	primary_hdu.header.update( 'has_meta', True )
	primary_hdu.header.update( 'model', 'M05', comment='meta data' )
	primary_hdu.header.update( 'met', met, comment='meta data' )
	primary_hdu.header.update( 'imf', imf, comment='meta data' )
	primary_hdu.header.update( 'sfh', sfh, comment='meta data' )
	if sfh == 'Exponential': primary_hdu.header.update( 'tau', tau, comment='meta data' )

# store the list of frequencies in a table
vs_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(name='vs', array=vs[sinds], format='D', unit='hertz')]))
# the list of ages
cols = [pyfits.Column(name='ages', array=ages, format='D', unit='years')]
# and masses
if filemass: cols.append( pyfits.Column(name='masses', array=masses, format='D', unit='m_sun') )
ages_hdu = pyfits.new_table(pyfits.ColDefs( cols ))
if filemass: ages_hdu.header.update( 'has_mass', True )

# make the fits file in memory
hdulist = pyfits.HDUList( [primary_hdu,vs_hdu,ages_hdu] )
# and write it out
hdulist.writeto( fileout, clobber=True )