#!/usr/bin/python

import glob,re,sys,math,pyfits
import numpy as np
import utils

if len( sys.argv ) < 2:
	print '\nconvert basti SSP models to ez_gal fits format'
	print 'Run in directory with SED models for one metallicity'
	print 'Usage: convert_basti.py ez_gal.ascii\n'
	sys.exit(2)

fileout = sys.argv[1]

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
	tau = parts[sfh_index+1] if sfh == 'exp' else ''
# metallicity
if parts.count( 'z' ):
	met = parts[ parts.index( 'z' ) + 1 ]
# imf
for (check,val) in zip( ['krou','salp','chab'], ['Kroupa', 'Salpeter', 'Chabrier'] ):
	if parts.count( check ):
		imf = val
		break
if parts.count( 'n' ):
	n = parts[ parts.index( 'n' ) + 1 ]
ae = False
if parts.count( 'ae' ): ae = True

# does the file with masses exist?
has_masses = False
mass_file = glob.glob( 'MLR*.txt' )
if len( mass_file ):
	# read it in!
	print 'Loading masses from %s' % mass_file[0]
	data = utils.rascii( mass_file[0], silent=True )
	masses = data[:,10:14].sum( axis=1 )
	has_masses = True

files = glob.glob( 'SPEC*agb*' )
nages = len( files )
ages = []

for (i,file) in enumerate(files):

	ls = []
	this = []

	# extract the age from the filename and convert to years
	m = re.search( 't60*(\d+)$', file )
	ages.append( int( m.group(1) )*1e6 )

	# read in this file
	fp = open( file, 'r' )
	for line in fp:
		parts = line.strip().split()
		ls.append( float( parts[0].strip() ) )
		this.append( float( parts[1].strip() ) )

	if i == 0:
		# if this is the first file, generate the data table
		nls = len( ls )
		seds = np.empty( (nls,nages) )

	# convert to ergs/s/angstrom
	seds[:,i] = np.array( this )/4.3607e-33/1e10

# convert to numpy
ages = np.array( ages )
ls = np.array( ls )*10.0

# make sure we are sorted in age
sinds = ages.argsort()
ages = ages[sinds]
seds = seds[:,sinds]

# speed of light
c = utils.convert_length( utils.c, incoming='m', outgoing='a' )

# convert from angstroms to hertz
vs = c/ls
# convert from ergs/s/A to ergs/s/Hz
seds *= ls.reshape( (ls.size,1) )**2.0/c
# and now from ergs/s/Hz to ergs/s/Hz/cm^2.0
seds /= (4.0*math.pi*utils.convert_length( 10, incoming='pc', outgoing='cm' )**2.0)

# sort in frequency space
sinds = vs.argsort()

# generate fits frame with sed in it
primary_hdu = pyfits.PrimaryHDU(seds[sinds,:])
primary_hdu.header.update( 'units', 'ergs/s/cm^2/Hz' )
primary_hdu.header.update( 'has_seds', True )
primary_hdu.header.update( 'nfilters', 0 )
primary_hdu.header.update( 'nzfs', 0 )

# store meta data
if sfh and met and imf:
	primary_hdu.header.update( 'has_meta', True )
	primary_hdu.header.update( 'model', 'BaSTI', comment='meta data' )
	primary_hdu.header.update( 'met', met, comment='meta data' )
	primary_hdu.header.update( 'imf', imf, comment='meta data' )
	primary_hdu.header.update( 'sfh', sfh, comment='meta data' )
	if sfh == 'Exponential': primary_hdu.header.update( 'tau', tau, comment='meta data' )
	primary_hdu.header.update( 'n', n, comment='meta data' )
	primary_hdu.header.update( 'ae', ae, comment='meta data' )

# store the list of frequencies in a table
vs_hdu = pyfits.new_table(pyfits.ColDefs([pyfits.Column(name='vs', array=vs[sinds], format='D', unit='hertz')]))
vs_hdu.header.update( 'units', 'hertz' )
# and the list of ages
cols = [pyfits.Column(name='ages', array=ages, format='D', unit='years')]
# and masses
if has_masses: cols.append( pyfits.Column(name='masses', array=masses, format='D', unit='m_sun') )
ages_hdu = pyfits.new_table(pyfits.ColDefs( cols ))
if has_masses: ages_hdu.header.update( 'has_mass', True )

# make the fits file in memory
hdulist = pyfits.HDUList( [primary_hdu,vs_hdu,ages_hdu] )
# and write it out
hdulist.writeto( fileout, clobber=True )