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

class DuplicateIndex(Exception):
	def __init__(self, message):
		self.message = 'index must be unique, while {} have duplicates'.format(message)

	def __str__(self):
		return self.message
