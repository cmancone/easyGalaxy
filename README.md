# EzGal

EzGal calculates observables (apparent magnitude, absolute magnitude, k-corrections, e-corrections, masses, mass-to-light-ratios, and distance moduli) of stellar populations as a function of redshift, given the formation redshift.  It can also generate spectral-energy distributions (SEDs) of complex-stellar populations (CSPs) from simple-stellar populations (SSPs).  It takes as a primary input grids of SEDs as a function of age which can be read from plain text files, the Bruzual and Charlot binary ised files, and its own format stored in binary fits tables.  It can pre-grid observables as a function of redshift and formation redshift and store the results in its own fits files to perform quick lookups later.

## Install

*Coming Soon*

## Example Usage

```py
import ezgal
from pylab import *

# load an EzGal model file
model = ezgal.model( 'bc03_ssp_z_0.02.fits' )

# get a grid of redshifts out to the desired formation redshift
zf = 3
zs = model.get_zs( zf )

# plot mass-to-light ratio evolution versus redshift for three filters
plot( zs, model.get_rest_ml_ratios( zf, filters='sloan_i', zs=zs ), 'k-', label='Sloan i' )
plot( zs, model.get_rest_ml_ratios( zf, filters='ks', zs=zs ), 'r--', label='2MASS Ks' )
plot( zs, model.get_rest_ml_ratios( zf, filters='ch2', zs=zs ), 'b:', label='IRAC ch2' )

# and set labels for the plot
xlabel( 'z' )
ylabel( 'M/L Ratio' )

# how about a legend?
legend()

# all done
show()
```

## API

Full API documentation can be viewed online in [html](http://www.baryons.org/ezgal/manual/) and [pdf](http://www.baryons.org/ezgal/manual.pdf) formats.

## Contributors

[Conor Mancone](https://github.com/cmancone/) and Anthony Gonzalez

## License

MIT