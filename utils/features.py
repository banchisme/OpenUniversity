r"""tools to build features"""
from utils import exceptions


class FeatureContainer:
	"""container class that hold all the individual features"""
	def __init__(self):
		self.features = {}

	def add(self, feature_name, feature):
		if feature_name in self.features:
			raise exceptions.FeatureAlreadyDefinedException(feature_name)
		else:
			self._sanity_check(feature)
			self.features[feature_name] = feature

	def get(self, feature_name):
		if feature_name in self.features:
			return self.features[feature_name].copy()
		else:
			raise exceptions.FeatureNotDefinedException(feature_name)

	def update(self, feature_name, feature):
		if feature_name in self.features:
			self.features[feature_name] = feature
		else:
			raise exceptions.FeatureNotDefinedException(feature_name)

	def pop(self, feature_name):
		if feature_name in self.features:
			return self.features.pop(feature_name)
		else:
			raise exceptions.FeatureNotDefinedException(feature_name)

	def delete(self, feature_name):
		self.pop(feature_name)

	def _sanitiy_check(self, feature):
		r"""minimum sanity check"""
		pass

	def get_feature_names(self):
		return list(self.features.keys())

	def merge(self, feature_names):
		r"""merge features
			Argument:
				feature_names (list): a list of feature names to be merged, when it is [],
					all features will be merged
			Return:
				merged feature (Feature)
		"""
		if len(feature_names) == 0:
			feature_names = self.get_feature_names()
		elif len(feature_names) == 1:
			return self.get(feature_names[0])

		feature = self.get(feature_names[0])
		for feature_name in feature_names[1:]:
			feature = feature.merge(self.get(feature_name))

		return feature


class Feature:
	r"""wrapper class for individual feature"""
	def __init__(self, key, val):
		r"""feature constructure
			Argument:
				key (str): feature_name
				val (pd.DataFrame): feature value, pandas dataframe, indexed
		"""
		self.name = key
		self._sanity_check(val)
		self.data = val.copy()

	def _sanity_check(self, val):
		pass

	def copy(self):
		r"""create a copy of the feature"""
		return Feature(key=self.name, val=self.data)

	def merge(self, f2, how='left', **kwargs):
		r"""merge with another feature
			Argument:
				f2 (Feature): a feature to be merged
				how (str): 'left', 'right', 'inner', 'outer'
				kwargs: kwargs to pass to the pd.DataFrame.merge function
			Return:
				merged feature (based on f1.index and f2.index)
		"""
		merged_name = self.name + '+' + f2.name
		merged_data = self.data.merge(f2.data, left_index=True, right_index=True, how=how, **kwargs)
		return Feature(merged_name, merged_data)

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, new_name):
		self.__name = new_name

	@property
	def data(self):
		return self.__data.copy()

	@data.setter
	def data(self, new_data):
		"""you should be able to change the data through the setter"""
		if self.data is None:
			self.__data = new_data

