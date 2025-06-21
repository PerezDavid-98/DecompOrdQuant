import numpy as np
import pandas as pd
import os

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDy
from quantificationlib.multiclass.energy import EDy
from quantificationlib.metrics.ordinal import emd

from quantificationlib.estimators.frank_and_hall import FrankAndHallMonotoneClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from quantificationlib.bag_generator import PriorShift_BagGenerator

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

# Añadido para elminar los warning del solver del estimador.
import warnings
warnings.filterwarnings("ignore")


# MAIN

dataset_files = [
         'data\\UCI\\ESL.csv',
         'data\\UCI\\LEV.csv',
         'data\\UCI\\SWD.csv',
         'data\\UCI\\bostonhousing.ord_chu.csv',
         'data\\UCI\\abalone.ord_chu.csv'
    ]

normalization = True
nbags = 200  # 50 * n_classes
seed = 42

estimator_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
estimator = LogisticRegression(random_state=seed, max_iter=1000, solver="liblinear")

nreps = 10

methods_names = ['AC', 'PAC', 'EM', 'EDy', 'HDy', 'FH-AC', 'FH-PAC', 'FH-EM', 'FH-EDy', 'FH-HDy']

est_name = 'LR'

# to store all the results
results = np.zeros((len(methods_names), len(dataset_files)))

def load_data(dfile, seed):
    df = pd.read_csv(dfile, sep=';', header=0)
    # cols = df.values.shape[1]
    # x = df.values[:, 0:cols - 1]
    # y = df.values[:, cols - 1]
    x = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    if normalization:
        # print('normalizying')
        x_train, x_test = normalize(x_train, x_test)
    return x_train, x_test, y_train, y_test


def normalize(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train.astype(np.float64))
    x_train = scaler.transform(x_train.astype(np.float64))
    x_test = scaler.transform(x_test.astype(np.float64))
    return x_train, x_test


def train_on_a_dataset(file_num, dfile, rep, current_seed):

    # generating training-test partition
    x_train, x_test, y_train, y_test = load_data(dfile, current_seed)

    #  n_classes = len(np.unique(y_train))
    n_bags_dataset = nbags  # * n_classes

    print('*** Training over {}, rep {}, seed {}, bags {}'.format(dataset_files[file_num], rep, current_seed, n_bags_dataset))

    # binary estimator

    skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                        scoring=make_scorer(geometric_mean_score), n_jobs=None)
    gs.fit(x_train, y_train)
    best_lr = gs.best_estimator_

    print(best_lr)

    
    # frank and hall estimator
    estimator_fh = FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=None)

    ac = AC(estimator_train=estimator_fh, estimator_test=estimator_fh, distance='L2')
    pac = PAC(estimator_train=estimator_fh, estimator_test=estimator_fh, distance='L2')
    em = EM(estimator_train=estimator_fh, estimator_test=estimator_fh)
    hdy = HDy(estimator_train=estimator_fh, estimator_test=estimator_fh, n_bins=8)
    edy = EDy(estimator_train=estimator_fh, estimator_test=estimator_fh)


    fh_ac = FrankAndHallQuantifier(quantifier=AC(),
                                    estimator_train=estimator_fh, estimator_test=estimator_fh)
    fh_pac = FrankAndHallQuantifier(quantifier=PAC(),
                                    estimator_train=estimator_fh, estimator_test=estimator_fh)

    fh_em = FrankAndHallQuantifier(quantifier=EM(),
                                    estimator_train=estimator_fh, estimator_test=estimator_fh)

    fh_hdy = FrankAndHallQuantifier(quantifier=HDy(n_bins=8),
                                    estimator_train=estimator_fh, estimator_test=estimator_fh)
    fh_edy = FrankAndHallQuantifier(quantifier=EDy(),
                                    estimator_train=estimator_fh, estimator_test=estimator_fh)
    

    #methods
    methods = [ac, pac, em, edy, hdy, fh_ac, fh_pac, fh_em, fh_edy, fh_hdy]

    print("Fitting models: \n")
    for nmethod, method in enumerate(methods):
        method.fit(X=x_train, y=y_train)
        print(f"Fitting method: {methods_names[nmethod]}")

    print('Test', end=' ')
        #  Testing bags
    bag_generator = PriorShift_BagGenerator(n_bags=nbags, bag_size=len(x_test),
                                            min_prevalence=None, random_state=seed + rep + 10)

    prev_true, indexes = bag_generator.generate_bags(x_test, y_test)


    for n_bag in range(nbags):
        print('Testing Bag #%d' % n_bag, flush=True)
        for nmethod, method in enumerate(methods):
            print(f"Testing method: {methods_names[nmethod]}")
            print(nmethod+1, end='')
            p_predicted = method.predict(X=x_test[indexes[:, n_bag], :])

            error = emd(prev_true[:, n_bag], p_predicted)

            results[nmethod, file_num] = results[nmethod, file_num] + error

    return None


dataset_names = [os.path.split(name)[-1][:-4] for name in dataset_files]
print('{} datasets to be processed'.format(len(dataset_files)))


for file_num, dfile in enumerate(dataset_files):
    for rep in range(nreps):
        train_on_a_dataset(file_num, dfile, rep + 1, seed)


results = results / (nreps * nbags)
name_file = "avg-datasets-UCI.csv"
file_avg = open("data\\" + name_file, 'w')
file_avg.write('Dataset,')
for index, m in enumerate(methods_names):
    file_avg.write('%s,' % m)
file_avg.write('\n')
for i in range(len(dataset_files)):
    file_avg.write('%s,' % dataset_names[i])
    for index, m in enumerate(methods_names):
        file_avg.write('%.5f,' %  results[index, i])
    file_avg.write('\n')

file_avg.close()

