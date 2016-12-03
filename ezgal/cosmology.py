import sys, math
import scipy.integrate as integrate
from scipy import inf
import numpy as num

__author__ = 'becker@astro.washington.edu (Andrew Becker)'

# becker@astro.washington.edu

# Many of these adapted from
# astro-ph/9905116

# README :
# Version 1.1 fixes bug in self.Tfunc (lookback time); thanks to Neil Crighton for catching it!
# Version 1.1.1 added unit conversions and changing cosmology at init (Conor Mancone, cmancone@ufl.edu)
# Version 1.1.2 added a couple extra functions of various utility (Conor Mancone)

VERSION = '1.1.2'

class Cosmology:
    lookup_tol = 1e-8
    # Note, all the distance units come from Dh, so this returns everything in meters
    
    def __init__(self, Om=0.272, Ol=0.728, h=0.704, w=-1):
        self.Om = Om
        self.Ol = Ol
        self.w  = w
        self.h  = h

        # CONSTANTS
        self.c       = 2.9979E5    # km/s
        self.G       = 6.67259E-11 # m^3 / kg / s^2
        self.Msun    = 1.98892E30  # kg
        self.pc      = 3.085677E16 # m

        # Functions for integration
        # Eqn 14 from Hogg, adding in 'w' to Ol.
        self.Efunc = lambda z, self=self : num.sqrt( (self.Om   * (1. + z)**3 +                 \
                                                       self.Ok() * (1. + z)**2 +                 \
                                                       self.Ol   * (1. + z)**(3 * (1. + self.w)) \
                                                       )**-1 )
        # Eqn 30
        self.Tfunc = lambda z, self=self : self.Efunc(z) / (1. + z)
                                           
                                           
                                           


    # Omega total
    def Otot(self):
        return self.Om + self.Ol

    # Curvature
    def Ok(self):
        return 1. - self.Om - self.Ol

    # Hubble constant, km / s / Mpc
    def H0(self):
        return 100 * self.h

    # Hubble constant at a particular epoch
    # Not sure if this is correct
    #def Hz(self, z):
    #    return self.H0() * (1. + z) * num.sqrt(1 + self.Otot() * z)

    # Got this from Jake
    def Hz(self, z):
        return self.H0 / self.Efunc(z)

    # Scale factor
    def a(self, z):
        return 1. / (1. + z)

    # Hubble distance, c / H0
    # Returns meters
    def Dh(self, meter=False, pc=False, kpc=False, mpc=False, cm=False):
        d  = self.c / self.H0()  # km / s / (km / s / Mpc) = Mpc
        d *= self.pc * 1e6       # m
        return d * self.lengthConversion(cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc)

    # Hubble time, 1 / H0
    # Returns seconds
    def Th(self, s=False, yr=False, myr=False, gyr=False):
        t  = 1. / self.H0()  # Mpc s / km
        t *= self.pc * 1e3
        return t * self.timeConversion(s=s, yr=yr, myr=myr, gyr=gyr)

    # Lookback time
    # Difference between the age of the Universe now and the age at z
    def Tl(self, z, s=False, yr=False, myr=False, gyr=False):
        return self.Th() * integrate.romberg(self.Tfunc, 0, z) * self.timeConversion(s=s, yr=yr, myr=myr, gyr=gyr)

    # Line of sight comoving distance
    # Remains constant with epoch if objects are in the Hubble flow
    def Dc(self, z, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        return self.Dh() * integrate.romberg(self.Efunc, 0, z) * self.lengthConversion(cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc)

    # Transverse comoving distance
    # At same redshift but separated by angle dtheta; Dm * dtheta is transverse comoving distance
    def Dm(self, z, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        Ok  = self.Ok()
        sOk = num.sqrt(num.abs(Ok))
        Dc  = self.Dc(z)
        Dh  = self.Dh()

        conversion = self.lengthConversion(cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc)

        if Ok > 0:
            return Dh / sOk * num.sinh(sOk * Dc / Dh) * conversion
        elif Ok == 0:
            return Dc * conversion
        else:
            return Dh / sOk * num.sin(sOk * Dc / Dh) * conversion

    # Angular diameter distance
    # Ratio of an objects physical transvserse size to its angular size in radians
    def Da(self, z, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        return self.Dm(z) / (1. + z) * self.lengthConversion(cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc)

    # Angular diameter distance between objects at 2 redshifts
    # Useful for gravitational lensing
    def Da2(self, z1, z2):
        # does not work for negative curvature
        assert(self.Ok()) >= 0

        # z1 < z2
        if (z2 < z1):
            foo = z1
            z1  = z2
            z2  = foo
        assert(z1 <= z2)

        Dm1 = self.Dm(z1)
        Dm2 = self.Dm(z2)
        Ok  = self.Ok()
        Dh  = self.Dh()

        return 1. / (1 + z2) * ( Dm2 * num.sqrt(1. + Ok * Dm1**2 / Dh**2) - Dm1 * num.sqrt(1. + Ok * Dm2**2 / Dh**2) )


    # Luminosity distance
    # Relationship between bolometric flux and bolometric luminosity
    def Dl(self, z, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        return (1. + z) * self.Dm(z) * self.lengthConversion(cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc)

    # Distance modulus
    # Recall that Dl is in m
    def DistMod(self, z):
        return 5. * num.log10(self.Dl(z) / self.pc / 10)

    # added 12/20/11 - Conor Mancone
    def Tu( self, s=False, yr=False, myr=False, gyr=False ):
        return self.Th() * integrate.quad(self.Tfunc, 0, inf)[0] * self.timeConversion(s=s, yr=yr, myr=myr, gyr=gyr)

    # added 12/16/11 - Conor Mancone
    # returns conversion from arcseconds to physical angular size
    def scale(self, z, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        return math.tan( 1.0/3600*math.pi/180 )*self.Da( z, cm=cm, meter=meter, pc=pc, kpc=kpc, mpc=mpc )

    # added 12/05/11 - Conor Mancone  Ages should be in years
    def GetZ(self, incoming_ages, zf):

        # deal with incoming scalar values
        is_scalar = False
        if len( num.asarray( incoming_ages ).shape ) == 0:
            incoming_ages = num.asarray( [incoming_ages] )
        else:
            incoming_ages = num.asarray( incoming_ages )

        # there is no cosmology routine to calculate redshift given formation redshift and age, so we must work the problem backwards.
        # Make a semi-regular grid of zs, calculate age given zf, and then interpolate on that.

        # Keep track of zfs we've looked up and their stored interpolation data
        if not hasattr( self, 'lookup_zfs' ):
            self.lookups = []
            self.lookup_zfs = num.array( [] )

        # if we have looked up this formation redshift before then figure out where it is in the lookups array
        fetched = False
        if len( self.lookup_zfs ):
            ind = num.abs( self.lookup_zfs - zf ).argmin()
            if num.abs( self.lookup_zfs - zf )[ind] < self.lookup_tol:
                ages = self.lookups[ind]['ages']
                zs = self.lookups[ind]['zs']
                fetched = True

        # if we haven't found the stored index then we have to look this one up now
        if not fetched:

            # make a list of redshifts/ages out to zf.
            # use finer sampling at low redshift.  The details aren't physically motivated - so sue me.
            zs = num.arange( 0.0, 0.01, 0.001 )
            if zf >= 0.01: zs = num.append( zs, num.arange( 0.01, num.min( [0.1, zf+1e-5] ), 0.005 ) )
            if zf >= 0.1:  zs = num.append( zs, num.arange( 0.1, num.min( [2.0, zf+1e-5] ), 0.025 ) )
            if zf >= 2.0:  zs = num.append( zs, num.arange( 2.0, zf+1e-5, 0.1 ) )

            # reverse so that age is monotonically increasing
            zs = zs[::-1]

            # now calculate ages for all those redshifts given formation redshift
            ages = num.array( [ self.Tl( zf, yr=True ) - self.Tl( z, yr=True ) for z in zs ] )

            # store reversed so age is monotonically increasing
            self.lookup_zfs = num.append( self.lookup_zfs, zf )
            self.lookups.append( {'zs': zs, 'ages': ages} )

        # now use the data in self.lookup_zfs[stored_ind] to do the lookup
        # return nan if the age is out of bounds (i.e. z < 0 or z > zf)
        outgoing_zs = num.zeros( incoming_ages.size )
        outgoing_zs[:] = num.nan
        m = (incoming_ages >= ages.min()) & (incoming_ages <= ages.max())
        if m.sum(): outgoing_zs[m] = num.interp( incoming_ages, ages, zs )

        # all done!
        return outgoing_zs

    # added 12/11/09 - Conor Mancone
    def timeConversion(self, s=False, yr=False, myr=False, gyr=False):
        # all times come in as seconds.  Convert accordingly
        if yr: return 1.0/(3600.0*24*365.25)
        elif myr: return 1.0/(3600.0*24*365.25*1e6)
        elif gyr: return 1.0/(3600.0*24*365.25*1e9)
        else: return 1.0

    # added 12/11/09 - Conor Mancone
    def lengthConversion(self, cm=False, meter=False, pc=False, kpc=False, mpc=False):
        # all lengths come in as meters.  Convert accordingly
        if cm: return 100.0
        elif pc: return 1.0/self.pc
        elif kpc: return 1.0/(self.pc*1e3)
        elif mpc: return 1.0/(self.pc*1e6)
        else: return 1.0

if __name__ == '__main__':
    
    c = Cosmology()

    if len(sys.argv) < 2:
        print 'Usage : cosmology.py z1 [z2]'
    z1 = float(sys.argv[1])

    print 'Cosmology : H0           =', c.H0()
    print 'Cosmology : Omega Matter =', c.Om
    print 'Cosmology : Omega Lambda =', c.Ol
    print ''
    
    print 'Hubble distance                %.2f Mpc' % (c.Dh() / c.pc / 1e6)
    print 'Hubble time                    %.2f Gyr' % (c.Th() / 3600 / 24 / 365.25 / 1e9)
    print ''
    
    print 'For z = %.2f:' % (z1)
    print 'Lookback time                  %.2f Gyr' % (c.Tl(z1) / 3600 / 24 / 365.25 / 1e9)
    print 'Scale Factor a                 %.2f'     % (c.a(z1))
    print 'Comoving L.O.S. Distance (w)   %.2f Mpc' % (c.Dc(z1) / c.pc / 1e6)
    print 'Angular diameter distance      %.2f Mpc' % (c.Da(z1) / c.pc / 1e6)
    print 'Luminosity distance            %.2f Mpc' % (c.Dl(z1) / c.pc / 1e6)
    print 'Distance modulus               %.2f mag' % (c.DistMod(z1))
