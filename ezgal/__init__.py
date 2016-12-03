import ezgal,utils,astro_filter,ezgal_light,wrapper,sfhs,weight,dusts

__all__ = ["model", "utils", "wrapper", "sfhs", "weight"]
__author__ = 'Conor Mancone, Anthony Gonzalez'
__email__ = 'cmancone@gmail.com'
__ver__ = '2.0'

ezgal = ezgal.ezgal
model = ezgal
astro_filter = astro_filter.astro_filter
ezgal_light = ezgal_light.ezgal_light
wrapper = wrapper.wrapper
weight = weight.weight

def interpolate( values, xs, models=None, key=None, return_wrapper=False ):
	""" models = ezgal.interpolate( values, xs, models, return_wrapper=False )
	
	or
	
	models = ezgal.interpolate( values, models, key=meta_key, return_wrapper=False )
	
	Interpolate between EzGal models and return new models.
	`models` is a list of EzGal model objects or filenames of EzGal compatible files.
	`xs` is the values of the models to be interpolated between and `values` is a list
	of values for the new models to be interpolated at.
	
	Alternatively you can ignore xs and specify the name of a meta key
	to use to build the interpolation grid.
	
	Returns a list of EzGal model objects or a single EzGal model if a scalar is passed
	for `values`.  Alternatively, set return_wrapper=True and it will return an ezgal wrapper
	object containing the fitted models objects.
	
	All model SEDs must have the same age/wavelength grid. """

	# what calling sequence was used?
	if models is None and key is not None:
		return wrapper( xs ).interpolate( key, values, return_wrapper=return_wrapper )

	# make sure we have everything we need...
	if len( models ) != len( xs ): raise ValueErrors( 'xs list has a different length than models list!' )

	# return interpolated models
	return wrapper( models, extra_data=xs, extra_name='interp' ).interpolate( 'interp', values, return_wrapper=return_wrapper )