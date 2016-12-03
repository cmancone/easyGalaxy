#!/usr/bin/python

import pyfits
import numpy as np
import scipy.interpolate

# speed of light in angstroms
c = 2.99792458e18
# conversion to d=10 pc from 1 AU
rat = (1.0/(3600*180/np.pi*10))**2.0

# open reference spectra
fits = pyfits.open( 'sun_reference_stis_001.fits' )
obs = fits[1].data
#fits = pyfits.open( 'sun_reference_red.fits' )
fits = pyfits.open( 'sun_kurucz93.fits' )
model = fits[1].data

# comparison region (a little broader for the observed spectrum because we will be interpolating it)
lmin = 15000
lmax = 25000
mo = (obs.field('wavelength') > (lmin-15) ) & (obs.field('wavelength') < (lmax+15))
mm = (model.field('wavelength') > lmin) & (model.field('wavelength') < lmax)

interp = scipy.interpolate.interp1d( obs.field('wavelength'), obs.field('flux') )
# interpolate observed flux at model points (scipy is being screwy and messes up if I pass it the whole array)
wm = np.where( mm )[0]
fluxobs = np.empty( wm.size )
for i in range( fluxobs.size ): fluxobs[i] = interp( model.field('wavelength')[wm[i]] )
# scale difference between observed and model solar spectrum
scale = np.mean( fluxobs/model.field('flux')[mm] )

# now combine the two spectra
m = model.field('wavelength') > obs.field('wavelength').max()
ls = np.append( obs.field('wavelength'), model.field('wavelength')[m] )
flux = np.append( obs.field('flux'), model.field('flux')[m]*scale )

# convert to Fv
fv = flux*ls**2.0/c*rat

# convert from angstroms to frequency
vs = c/ls

# sort in frequency space
sinds = vs.argsort()

# and save
tbhdu = pyfits.new_table( [pyfits.Column( name='freq', format='D', array=vs[sinds] ), pyfits.Column( name='flux', format='D', array=fv[sinds] )] )
tbhdu.writeto( 'solar_new.fits', clobber=True )