from unittest import TestCase
from gefs.cgef import *
from experiments.prep import get_data


class Test(TestCase):
    def test_dataset(self):
        D1 = DataSet()
        X = [[2, 4, 0], [1, 3, 0], [3, 3, 1], [2, 3, 1]]
        ncat = [4, 5, 2]
        D1.from_numpy(X, ncat)
        print(D1)

        X1, ncat1 = D1.to_numpy()
        D2 = DataSet()
        D2.from_numpy(X1, ncat1)
        print(D2)

        self.assertEqual(D1, D2)


    def test_random_classifier(self):
        X, ncat = get_data('authent')
        D = DataSet()
        D.from_numpy(X, ncat)

        classifier = RandomForestClassifier(n_estimators=100,
                                            min_samples_leaf=5,
                                            max_depth=1000,
                                            criterion="gini",
                                            sample_fraction=1,
                                            sample_technique="stratified",
                                            execution_mode="sequential",
                                            split_family="threshold-subset"
                                           )

        rf = classifier.fit(D, seed=123456)
        self.assertEqual(len(rf.trees()), 100)
