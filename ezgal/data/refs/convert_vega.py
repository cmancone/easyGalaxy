#!/usr/bin/python

import pyfits

files = {'vega.fits': 'alpha_lyr_stis_005.fits'}

# speed of light in angstroms
c = 2.99792458e18

for ( fileout, filein ) in files.iteritems():

	# open reference spectrum
	fits = pyfits.open( filein )

	# load
	ls = fits[1].data.field('wavelength')
	fl = fits[1].data.field('flux')

	# convert to Fv
	fv = fl*ls**2.0/c

	# convert from angstroms to frequency
	vs = c/ls

	# sort in frequency space
	sinds = vs.argsort()

	# and save
	tbhdu = pyfits.new_table( [pyfits.Column( name='freq', format='D', array=vs[sinds] ), pyfits.Column( name='flux', format='D', array=fv[sinds] )] )
	tbhdu.writeto( fileout, clobber=True )