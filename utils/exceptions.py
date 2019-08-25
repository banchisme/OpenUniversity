class FeatureNotDefinedException(Exception):
	def __init__(self, feature_name=''):
		self.message = 'feature "{}" is not defined'.format(feature_name)

	def __str__(self):
		return self.message


class FeatureAlreadyDefinedException(Exception):
	def __init__(self, feature_name=''):
		self.message = 'feature "{}" is already defined'.format(feature_name)

	def __str__(self):
		return self.message
