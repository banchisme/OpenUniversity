r"""tools to build features"""
from university import exceptions


class FeatureDict(dict):
	"""a container class that hold all the individual features"""

	def update(self, feature_dict, name_conflict='raise_error'):
		r"""
		union with another feature container
		:param container (feature container):
		:return:
		None, union inplace
		"""

		if name_conflict == 'raise_error':
			for feature_name, feature in feature_dict.itmes():
				if feature_name not in self:
					self[feature_name] = feature
				else:
					raise exceptions.FeatureAlreadyDefinedException(feature_name)
		else:
			raise NotImplementedError

	def merge(self, feature_names=[]):
		r"""merge features
			Argument:
				feature_names (list): a list of feature names to be merged, when it is [],
					all features will be merged
			Return:
				merged feature (Feature)
		"""
		if len(feature_names) == 0:
			feature_names = list(self.keys())
		elif len(feature_names) == 1:
			return self[feature_names[0]]

		feature = self[feature_names[0]].copy()
		for feature_name in feature_names[1:]:
			feature = feature.merge(self[feature_name])

		return feature


class Feature:
	r"""wrapper class for individual feature"""
	def __init__(self, key, val):
		r"""feature constructure
			Argument:
				key (str): feature_name
				val (pd.DataFrame): feature value, pandas dataframe, indexed
				mutable (bool): if mutable, feature could resign values
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
		return self._name

	@name.setter
	def name(self, new_name):
		self._name = new_name

	@property
	def data(self):
		return self._data.copy()

	@data.setter
	def data(self, new_data):
		"""you should not be able to change the data through the setter"""
		if hasattr(self, '_data') is False:
			self._data = new_data


