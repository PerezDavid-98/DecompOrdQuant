import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDy
from quantificationlib.multiclass.energy import EDy

from quantificationlib.estimators.frank_and_hall import FrankAndHallMonotoneClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from evaluate import  evaluate_submission
from data import ResultSubmission

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer

def load_training_set(dfile):
    data = np.genfromtxt(dfile, skip_header=1, delimiter=',')

    X = data[:, 1:]
    y = data[:, 0].astype(int)
    return X, y

def load_testing_bag(dfile):
    X = np.genfromtxt(dfile, skip_header=1, delimiter=',')
    return X

def absolute_error(prevs, prevs_hat):
    assert prevs.shape == prevs_hat.shape, 'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).mean(axis=-1)


def relative_absolute_error(p, p_hat, eps=None):
    def __smooth(prevs, epsilon):
        n_classes = prevs.shape[-1]
        return (prevs + epsilon) / (epsilon * n_classes + 1)

    p = __smooth(p, eps)
    p_hat = __smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)



def main(path, dataset, estimator_name, n_bags=10, bag_inicial=0, master_seed=2032):

    method_name = ['AC', 'PAC', 'EM', 'EDy', 'HDy', 'FH-AC', 'FH-PAC', 'FH-EM', 'FH-EDy', 'FH-HDy']
    estimator_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    n_classes = 5
    n_train = 100

    X_train = np.empty((0, 256))
    y_train = np.empty((0))
    y_train_binary = np.empty((0))

    # All of the train files are merged into a single file
    for bag in range(n_train):
        X_train_bag, y_train_bag = load_training_set(path + dataset + '\\public\\training_samples\\' + str(bag) + '.txt')

        X_train = np.append(X_train, X_train_bag, axis=0)
        y_train = np.append(y_train, y_train_bag, axis=0)

        y_train_temp = y_train_bag.copy()
        y_train_temp[y_train_temp < n_classes / 2 ] = -1
        y_train_temp[y_train_temp > n_classes / 2 ] = 1

        y_train_binary = np.append(y_train_binary, y_train_temp)
        


    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they check whether the estimator is already fitted (by a previous object) or not
    if estimator_name == 'LR':
        estimator = LogisticRegression(random_state=master_seed, max_iter=1000, solver="liblinear")

    elif estimator_name == 'CLLR':
        estimator = CalibratedClassifierCV(LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced'))
 
    else:
        raise ValueError('Unknwon estimator')
    
    skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=master_seed)
    gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                        scoring=make_scorer(geometric_mean_score), n_jobs=None)
    gs.fit(X_train, y_train_binary)
    best_lr = gs.best_estimator_

    estimator_train = FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=None)
    estimator_test = None # CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced'))

    print('Fitting Training Estimator')
    estimator_train.fit(X_train, y_train)
    print('Training Estimator fitted', flush=True)
    probs_train = estimator_train.predict_proba(X_train)

    #  predictions_train = None
    print('Prediction_train computed')

    print('Fitting Estimator Test')
    if estimator_test is None:
        estimator_test = estimator_train
    else:
        estimator_test.fit(X_train, y_train)
    print('Estimator test fitted')

    # AC
    ac = AC()
    ac.fit(X_train, y_train, predictions_train=probs_train)
    print('AC fitted')
    # FH_AC
    fh_ac = FrankAndHallQuantifier(quantifier=AC(),
                                    estimator_train=estimator_train, estimator_test=estimator_test)
    fh_ac.fit(X_train, y_train, predictions_train=probs_train)
    print('FH_AC fitted')

    # PAC
    pac = PAC()
    pac.fit(X_train, y_train, predictions_train=probs_train)
    print('PAC fitted')
    # FH_PAC
    fh_pac = FrankAndHallQuantifier(quantifier=PAC(),
                                    estimator_train=estimator_train, estimator_test=estimator_test)
    fh_pac.fit(X_train, y_train, predictions_train=probs_train)
    print('FH_PAC fitted')

    # EM
    em = EM()
    em.fit(X_train, y_train, predictions_train=probs_train)
    print('EM fitted')
    # FH_EM
    fh_em = FrankAndHallQuantifier(quantifier=EM(),
                                    estimator_train=estimator_train, estimator_test=estimator_test)
    fh_em.fit(X_train, y_train, predictions_train=probs_train)
    print('FH_EM fitted')

    # EDy
    edy = EDy()
    edy.fit(X_train, y_train, predictions_train=probs_train)
    print('EDy fitted')
    # FH_EDy
    fh_edy = FrankAndHallQuantifier(quantifier=EDy(),
                                    estimator_train=estimator_train, estimator_test=estimator_test)
    fh_edy.fit(X_train, y_train, predictions_train=probs_train)
    print('FH_EDy fitted')

    # HDY
    hdy = HDy(n_bins=4, bin_strategy='equal_width')
    hdy.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy fitted')
    # FH_HDy
    fh_hdy = FrankAndHallQuantifier(quantifier=HDy(),
                                    estimator_train=estimator_train, estimator_test=estimator_test)
    fh_hdy.fit(X_train, y_train, predictions_train=probs_train)
    print('FH_HDy fitted')
    
    
    methods = [ac, pac, em, edy, hdy, fh_ac, fh_pac, fh_em, fh_edy, fh_hdy]

    # prev_true = load_prevalences(path + dataset + "\\public\\dev_prevalences.txt")

    # results_mae = np.zeros((n_bags, len(method_name)))
    # results_rmae = np.zeros((n_bags, len(method_name)))
    # results_pred = np.zeros((n_bags, len(method_name)))
    
    prev_preds =  np.zeros((len(method_name), n_bags, 5))
    for n_bag in range(n_bags):
        print('Validation Bag #%d' % n_bag, flush=True)
        X_test = load_testing_bag(path + dataset + '\\public\\dev_samples\\' + str(bag_inicial + n_bag) + '.txt')

        probs_test = estimator_test.predict_proba(X_test)

        for index, method in enumerate(methods):
            prev_preds[index, n_bag, :] = method.predict(X=None, predictions_test=probs_test)
        
    prev_preds[ prev_preds < 0 ] = 0 
    prev_preds[ prev_preds > 1 ] = 1 


    print('Saving and Computing results')    
    #  printing and saving results
    filename = path + r'\T3_preds\dev_preds\Validation-' + dataset + '-' + estimator_name
    file_avg_emd = open(filename + str(n_bags) + '_emd_avg.txt', 'w')


    prev_true = ResultSubmission.load(path + dataset + "\\public\\dev_prevalences_" + str(n_bags) + ".txt")

    for n_method, method in enumerate(method_name):    
        name_file_preds = path + r'\T3_preds\dev_preds\Validation-' + dataset + '-' + method + '-' + estimator_name + '-values-' + str(n_bags) + '-.txt'
        file_preds = open(name_file_preds, 'w')
        file_preds.write('id')
        for i in range(n_classes):
            file_preds.write(',%d' % (i))
        file_preds.write('\n')
        
        for n_bag in range(n_bags):
            file_preds.write('%d,' % (n_bag))
            prev_preds[n_method, n_bag, :].tofile(file_preds, sep=',')
            file_preds.write('\n')
        file_preds.close()    
 
        prev_preds_method = ResultSubmission.load(name_file_preds)

        emd_score = evaluate_submission(prev_true, prev_preds_method, n_bags, measure='emd', average=False)

        file_avg_emd.write('%s,' % method)
        file_avg_emd.write('%.5f\n' % emd_score.mean())
    
    file_avg_emd.close()


if __name__ == '__main__':
    values = [50, 100, 200, 500, 700, 1000]

    for value in values:
        main(path='data', dataset='T3',
            estimator_name='LR', n_bags=value, bag_inicial=0, master_seed=2032)
