import ezgal

class weight(object):
	""" ezgal.weight class
	
	Used to apply weights to EzGal model objects through multiplication """

	weight = 1
	ezgal_type = ''

	def __init__( self, weight ):
		self.weight = float( weight )
		self.ezgal_type = type( ezgal.ezgal( skip_load=True ) )

	def __mul__( self, obj ):
		if type( obj ) == type( self ):
			return weight( self.weight*obj.weight )
		elif type( obj ) == self.ezgal_type:
			return obj.weight( self.weight )
		else:
			return weight( self.weight*obj )

	def __imul__( self, obj ):
		if type( obj ) == type( self ):
			self.weight *= obj.weight
		elif type( obj ) == self.ezgal_type:
			self.weight *= obj.model_weight
		else:
			self.weight *= obj
		return self