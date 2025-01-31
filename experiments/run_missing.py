#!/usr/bin/env python3

import argparse
import gc
from tqdm import tqdm

import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
from scipy import stats
from sklearn.metrics import accuracy_score

from gefs.learning import LearnSPN
from gefs.trees import Tree, RandomForest

from prep import get_data, learncats, get_stats, normalize_data, standardize_data
from knn_imputer import KNNImputer
from simple_imputer import SimpleImputer
from miss_forest import MissForest
from gefs.cgef import DataSet, RandomForestClassifier
from cgeflib import build_generative_forest


# Auxiliary functions
def str2bool(v):
    """ Converts a string to a boolean value. """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='wine',
    )

    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    parser.add_argument(
        '--n_folds', '-f',
        type=int,
        default=5,
    )

    parser.add_argument(
        '--n_estimators', '-e',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--msl', '-m',
        nargs='+',
        type=int,
        default=[1],
    )

    parser.add_argument(
        '--lspn', '-l',
        type=str2bool,
        default='false',
    )

    FLAGS, unparsed = parser.parse_known_args()

    data, ncat = get_data(FLAGS.dataset)

    # min_sample_leaf is the minimum number of samples at leaves
    # Define which values of min_sample_leaf to test
    min_sample_leaves = FLAGS.msl
    min_sample_leaves = [msl for msl in min_sample_leaves if msl < data.shape[0]/10]

    meanspath = os.path.join('missing', FLAGS.dataset + '_mf_means.csv')
    cispath = os.path.join('missing', FLAGS.dataset + '_mf_cis.csv')
    Path('missing').mkdir(parents=True, exist_ok=True)

    if FLAGS.lspn:
        methods = ['Friedman', 'Mean', 'Surr', 'KNN', 'MissForest', 'GeFp', 'GeF', 'GeFp(LSPN)', 'GeF(LSPN)']
    else:
        methods = ['Friedman', 'Mean', 'Surr', 'KNN', 'MissForest', 'GeFp', 'GeF']

    df_all = pd.DataFrame()

    np.seterr(invalid='raise')
    completed_runs = 0
    for run in FLAGS.runs:
        print('####### DATASET: ', FLAGS.dataset, " with shape ", data.shape)
        print('####### RUN: ', run)
        print('####### ', FLAGS.n_folds, ' folds')

        np.random.seed(run)  # Easy way to reproduce the folds
        folds = np.zeros(data.shape[0])
        for c in np.unique(data[ :, -1]):
            nn = np.sum(data[ :, -1] == c)
            ind = np.tile(np.arange(FLAGS.n_folds), int(np.ceil(nn/FLAGS.n_folds)))[:nn]
            folds[data[:, -1] == c] = np.random.choice(ind, nn, replace=False)

        for min_samples_leaf in min_sample_leaves:
            print('        min samples: ', min_samples_leaf)
            for fold in range(FLAGS.n_folds):
                print('####### Fold: ', fold)
                train_data = data[np.where(folds!=fold)[0], :]
                test_data = data[np.where(folds==fold)[0], :]

                # Standardize train data only
                _, maxv, minv, mean, std = get_stats(train_data, ncat)
                train_data = standardize_data(train_data, mean, std)
                test_data = standardize_data(test_data, mean, std)

                X_train, X_test = train_data[:, :-1], test_data[:, :-1]
                y_train, y_test = train_data[:, -1], test_data[:, -1]

                imputer = SimpleImputer(ncat=ncat[:-1], method='mean').fit(X_train)

                knn_imputer = KNNImputer(ncat=ncat[:-1], n_neighbors=7).fit(X_train)

                cat_vars = np.where(ncat[:-1]>1)[0]
                if len(cat_vars) == 0:
                    cat_vars = None
                forest_imputer = MissForest(random_state=run).fit(X_train, cat_vars=cat_vars)

                np.random.seed(run)
                print('            Training')
                rf = RandomForest(n_estimators=FLAGS.n_estimators, ncat=ncat, min_samples_leaf=min_samples_leaf, surrogate=True, random_state=run)
                rf.fit(X_train, y_train)

                print('            Converting to GeF')
                gef = rf.topc()
                gef.maxv, gef.minv = maxv, minv

##################################################################################################
                print('            Training (cgef)')
                D_train = DataSet()
                D_train.from_numpy(train_data, ncat)
                D_test = DataSet()
                D_test.from_numpy(test_data, ncat)
                classifier = RandomForestClassifier(n_estimators=FLAGS.n_estimators,
                                                    min_samples_leaf=min_samples_leaf,
                                                    max_depth=1000,
                                                    criterion="gini",
                                                    sample_fraction=1,
                                                    sample_technique="stratified",
                                                    execution_mode="parallel",
                                                    split_family="threshold-subset"
                                                   )
                rf_train = classifier.fit(D_train, seed=12345)

                print('            Converting to GeF (cgef)')
                pc_train = build_generative_forest(rf_train, D_train)

##################################################################################################

                if FLAGS.lspn:
                    print('            Converting to GeF(LSPN)')
                    gef_lspn = rf.topc(learnspn=30) # 30 is the number of samples required to fit LearnSPN
                    gef_lspn.maxv, gef_lspn.minv = maxv, minv

                print("            Inference")
                for i in tqdm(np.arange(0, 1., 0.1)):
                    df = pd.DataFrame()
                    if i == 0.:
                        gefe = gef.classify_avg(X_test, classcol=data.shape[1]-1, return_prob=False)
                        nomiss = rf.predict(X_test, vote=False)
                        gefp = gef.classify(X_test, classcol=data.shape[1]-1, return_prob=False)
                        # All other methods are the same for complete data
                        fried = nomiss
                        imp_mean = nomiss
                        imp_knn = nomiss
                        imp_mf = nomiss
                        surr = nomiss
                        if FLAGS.lspn:
                            l_nomiss = gef_lspn.classify_avg_lspn(X_test, classcol=data.shape[1]-1, return_prob=False)
                            gefp_lspn = gef_lspn.classify_lspn(X_test, classcol=data.shape[1]-1, return_prob=False)
                            gefe_lspn = l_nomiss
                        print("Base Accuracy: ", accuracy_score(y_test, nomiss))

                    else:
                        # Sets random values to NaN
                        np.random.seed(run)
                        missing_mask = np.full(X_test.size, False)
                        missing_mask[:int(i * X_test.size)] = True
                        np.random.shuffle(missing_mask)
                        missing_mask = missing_mask.astype(bool)

                        X_test_miss = X_test.copy()
                        X_test_miss.ravel()[missing_mask] = np.nan

                        fried = gef.classify_avg(X_test_miss, classcol=data.shape[1]-1, naive=True, return_prob=False)

                        surr = rf.predict(X_test_miss, vote=False)

                        X_test_miss_1 = imputer.transform(X_test_miss)
                        assert np.sum(np.isnan(X_test_miss_1)) == 0, "Bug in the simple imputation method."
                        # imp_mean = gef.classify_avg(X_test_miss_1, classcol=data.shape[1]-1, return_prob=False)
                        imp_mean = rf.predict(X_test_miss_1, vote=False)

                        X_test_miss_2 = knn_imputer.transform(X_test_miss)
                        assert np.sum(np.isnan(X_test_miss_2)) == 0, "Bug in the KNN imputation method."
                        # imp_knn = gef.classify_avg(X_test_miss_2, classcol=data.shape[1]-1, return_prob=False)
                        imp_knn = rf.predict(X_test_miss_2, vote=False)

                        X_test_miss_3 = forest_imputer.transform(X_test_miss)
                        assert np.sum(np.isnan(X_test_miss_3)) == 0, "Bug in the MissForest imputation method."
                        # imp_mf = gef.classify_avg(X_test_miss_3, classcol=data.shape[1]-1, return_prob=False)
                        imp_mf = rf.predict(X_test_miss_3, vote=False)

                        gefp = gef.classify(X_test_miss, classcol=data.shape[1]-1, return_prob=False)
                        gefe = gef.classify_avg(X_test_miss, classcol=data.shape[1]-1, return_prob=False)

                        if FLAGS.lspn:
                            gefp_lspn = gef_lspn.classify_lspn(X_test_miss, classcol=data.shape[1]-1, return_prob=False)
                            gefe_lspn = gef_lspn.classify_avg_lspn(X_test_miss, classcol=data.shape[1]-1, return_prob=False)

                    df['Friedman'] = [accuracy_score(y_test, fried)]
                    df['Mean'] = [accuracy_score(y_test, imp_mean)]
                    df['Surr'] = [accuracy_score(y_test, surr)]
                    df['KNN'] = [accuracy_score(y_test, imp_knn)]
                    df['MissForest'] = [accuracy_score(y_test, imp_mf)]
                    df['GeFp'] = [accuracy_score(y_test, gefp)]
                    df['GeF'] = [accuracy_score(y_test, gefe)]
                    if FLAGS.lspn:
                        df['GeFp(LSPN)'] = [accuracy_score(y_test, gefp_lspn)]
                        df['GeF(LSPN)'] = [accuracy_score(y_test, gefe_lspn)]

                    df['Missing Test'] = [np.round(i, 2)]
                    df['Run'] = [run]
                    df['Fold'] = [fold]
                    df['Number of estimators'] = [FLAGS.n_estimators]
                    df['Min Samples'] = [min_samples_leaf]

                    # if file does not exist write header
                    df_all = pd.concat((df_all, df))

                rf.delete()
                del rf
                gef.delete()
                del gef
                if FLAGS.lspn:
                    gef_lspn.delete()
                    del gef_lspn
                gc.collect()

        completed_runs += 1

        conf = 0.95
        if len(FLAGS.runs) == 1:
            n_tests = FLAGS.n_folds
            means = df_all.groupby('Missing Test').mean().reset_index()[['Missing Test'] + methods]
            cis = df_all.groupby('Missing Test').std().reset_index()[['Missing Test'] + methods]
            cis[methods] = cis[methods]/np.sqrt(n_tests) * stats.t.ppf((1 + conf) / 2, n_tests - 1)
        else:
            n_tests = completed_runs # len(FLAGS.runs)
            df_all = df_all.groupby(['Run', 'Missing Test']).mean().reset_index()
            means = df_all.groupby('Missing Test').mean().reset_index()[['Missing Test'] + methods]
            cis = df_all.groupby('Missing Test').std().reset_index()[['Missing Test'] + methods]
            cis[methods] = cis[methods]/np.sqrt(n_tests) * stats.t.ppf((1 + conf) / 2, n_tests - 1)

        means.to_csv(meanspath, index=False)
        cis.to_csv(cispath, index=False)
