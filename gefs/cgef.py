import math
import numpy as np
import random
import cgeflib


class DataSet(cgeflib.DataSet):
    def __init__(self, X=None, ncat=None, features=None):
        cgeflib.DataSet.__init__(self)  # TODO: why does super.__init__(self) fail?
        if X:
            self.X = X
            self.ncat = ncat
        if features:
            self.features = features

    def from_numpy(self, X, ncat, features=None):
        X1 = cgeflib.DoubleMatrix([X[i] for i in range(len(X))])
        self.X = X1
        self.category_counts = ncat
        if features:
            self.features = features

    def to_numpy(self):
        n = self.row_count()
        X = self.X
        X1 = np.array([X[i] for i in range(n)])
        return X1, self.category_counts


class RandomForestClassifier(object):
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = "gini",
                 max_depth: int = None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="auto",
                 sample_fraction=1,
                 sample_technique: str = "stratified",
                 execution_mode: str = "sequential",
                 split_family=""):
        """
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy", "misclassification"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for and "misclassification" the information gain.
        Note: this parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    sample_fraction : float, default="1", `0 < sample_fraction <= 1`
        The number of samples that is selected for a fit, expressed as a fraction of the size of the dataset.

    sample_technique : {"without-replacement", "with-replacement", "stratified"}, default="stratified"
        The technique used to select samples from the dataset.

        - If "without-replacement" the selected samples are unique
        - If "with-replacement" the selected samples may contain duplicates
        - If "stratified" the samples are partitioned according to their class. For each partition approximately the
        same fraction of samples is selected, and duplicates are possible

    execution_mode : {"sequential", "parallel"}, default="sequential"
        The execution mode used for the \c fit algorithm. Supported values are "sequential" to let the algorithm use
        a single core, and "parallel" to allow the algorithm to use multiple cores.

    split_family : {"threshold", "threshold-single", "threshold-subset"}, default="threshold"
        The family of splits that is considered when splitting a node.

        - If "threshold" only threshold splits are considered
        - If "threshold-single" both threshold splits and single splits are considered
        - If "threshold-subset" both threshold splits and subset splits are considered
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.sample_fraction = sample_fraction
        self.sample_technique = sample_technique
        self.execution_mode = execution_mode
        self.split_family = split_family


    def _compute_max_features(self, D: DataSet):
        n_features = D.feature_count()
        max_features = self.max_features
        if isinstance(max_features, int):
            return max_features
        elif isinstance(max_features, float):
            return round(max_features * n_features)
        elif max_features == "auto":
            return int(math.sqrt(n_features))
        elif max_features == "sqrt":
            return int(math.sqrt(n_features))
        elif max_features == "log2":
            return math.log2(n_features)
        elif not max_features:
            return n_features
        raise RuntimeError('unexpected value of max_features: {}'.format(max_features))


    def fit(self, D: DataSet, seed: int = random.randint(-214783648, 2147483647)) -> cgeflib.RandomForest:
        """Build a forest of trees from the data set D."""
        tree_options = cgeflib.DecisionTreeOptions()
        tree_options.imp_measure = cgeflib.parse_impurity_measure(self.criterion)
        tree_options.max_features = self._compute_max_features(D)
        tree_options.support_missing_values = tree_options.support_missing_values or D.has_missing_values()
        tree_options.min_samples_leaf = self.min_samples_leaf

        forest_options = cgeflib.RandomForestOptions()
        forest_options.forest_size = self.n_estimators
        forest_options.sample_fraction = self.sample_fraction
        forest_options.sample_technique = cgeflib.parse_sample_technique(self.sample_technique)

        sequential = self.execution_mode == "sequential"
        n = D.row_count()
        I = range(n)

        return cgeflib.learn_random_forest(D, I, forest_options, tree_options, self.split_family, sequential, seed)

