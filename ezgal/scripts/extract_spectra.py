#!/usr/bin/python

import ezgal,sys
import numpy as np

# get call parameters from python
model_file = sys.argv[1]	# input file
output_file = sys.argv[2]	# output file

# list of ages
ages = np.asarray( [ float( val ) for val in sys.argv[3:] ] )
nages = len( ages )

# load the model file into EzGal
model = ezgal.model( model_file )

# and get the list of wavelengths
ls = model.ls

# interpolate to get the mass at each age
if model.has_masses:
	masses = np.interp( ages, model.ages/1e9, model.masses )

# now build an array to hold all the data which we will print to the output file
seds = np.empty( (ls.size,nages+1) )

# store the wavelengths in the array
seds[:,0] = ls

# now loop through the age list, fetch the SED, and store in the output array
# while we are at it make a list of format strings for writing the data to the file
formats = ['%7d']
for (i, age) in enumerate( ages ):
	seds[:,i+1] = model.get_sed( age, units='Fl' )
	formats.append( '%12.6e' )

# now output the file
fh = open( output_file, 'wb' )

# first a header
fh.write( "# 1 Wavelengths (Angstroms)\n" )
for (i, age) in enumerate( ages ):
	header = "# %d t = %f gyrs" % (i+2,age)
	if model.has_masses:
		header += ", mass = %f" % masses[i]
	header += "\n"
	fh.write( header )

# then the data
for i in range( ls.size ):
	fh.write( ' '.join([format % val for format,val in zip(formats, seds[i,:])]) + "\n" )

# all done
fh.close()