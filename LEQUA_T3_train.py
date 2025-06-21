import numpy as np

import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics.pairwise import euclidean_distances

from evaluate import normalized_match_distance, eval_metric, evaluate_submission
from data import ResultSubmission

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC

from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df  import HDy, DFy
from quantificationlib.multiclass.energy import EDy
from quantificationlib.ordinal.pdf import PDFOrdinaly

from quantificationlib.estimators.ordinal_ddag import DDAGClassifier
from quantificationlib.estimators.frank_and_hall import FrankAndHallMonotoneClassifier, FrankAndHallTreeClassifier
from quantificationlib.estimators.cross_validation import CV_estimator

from quantificationlib.metrics.multiclass import l1, l2
from quantificationlib.metrics.ordinal import emd, emd_distances

from sklearn.metrics.pairwise import euclidean_distances

def load_training_set(dfile):
    data = np.genfromtxt(dfile, skip_header=1, delimiter=',')

    X = data[:, 1:]
    y = data[:, 0].astype(int)
    return X, y

def load_testing_bag(dfile):
    X = np.genfromtxt(dfile, skip_header=1, delimiter=',')
    return X

def load_prevalences(dfile):
    data = np.genfromtxt(dfile, skip_header=1, delimiter=',')

    prevalences = data[:, 1:]
    return prevalences

def eval_macro_nmd(true_prevs, pred_prevs, average=False):
    
    stars = np.arange(5)+1  # [1,2,3,4,5]

    # computes the average stars rate
    mean_true_stars = (stars * true_prevs).sum(axis=1)

    # bins results by average stars rate in [1,2), [2,3), [3,4), [4,5]
    bin_idx = np.digitize(mean_true_stars, bins=stars)

    errors = np.zeros(shape=len(stars)-1, dtype=float)
    for star in stars[:-1]:
        select = bin_idx==star
        bin_true = true_prevs[select]
        bin_pred = pred_prevs[select]
        errors[star-1] = eval_metric(bin_true, bin_pred, normalized_match_distance, average=True)

    if average:
       errors = errors.mean()
    
    return errors

def main(estimator_name='CLLR', model='', file_preds='../datasets/dev_prevalences.txt', n_bags=1000, bag_inicial=0, master_seed=2032):

    method_name = ['CC', 'PCC', 'AC', 'PAC', 'EM',
                   'HDy-4b', 'HDy-8b', 'HDy-16b',
                   'CDFy-100b-L1', 
                   'EDy_eu', 'EDy_emd',
                   'PDFy8-l2','PDFy16-l2','PDFy32-l2',
                   'PDFy8-emd','PDFy16-emd','PDFy32-emd',
                  ]

    print('Loading Training data')
    X_train, y_train = load_training_set('../datasets/training_data.txt')
    print('Training data loaded')

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    if model != '':
        print('Loading Training Estimator')
        estimator_train = joblib.load(model)
        print('Training Estimator loaded')
        estimator_test = None
    elif estimator_name == 'MonotoneLR':
        estimator = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None
    elif estimator_name == 'MonotoneCLLR':
        estimator = CalibratedClassifierCV(LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced'))
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None    
    elif estimator_name == 'LR':
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
        # estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_train = estimator
        estimator_test = None # LogisticRegression(C=0.01, class_weight='balanced')
    elif estimator_name == 'CLLR':
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = CalibratedClassifierCV(LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced'))
        # estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_train = estimator
        estimator_test = None # CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced'))
    elif estimator_name == 'OVRLR':
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = OneVsRestClassifier(estimator=LogisticRegression(C=0.01, class_weight='balanced'), n_jobs=-1)
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None # LogisticRegression(C=0.01, class_weight='balanced')
    elif estimator_name == 'OVRCLLR':
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = OneVsRestClassifier(estimator=CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced')), n_jobs=-1)
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None # CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced')) # LogisticRegression(C=0.01, class_weight='balanced')  
    elif estimator_name == 'CLRF':
        estimator = RandomForestClassifier(n_estimators=500, max_depth=2,random_state=master_seed)
        estimator_train = CalibratedClassifierCV(estimator)
        estimator_test = None
    elif estimator_name == 'RF':
        estimator = RandomForestClassifier(n_estimators=500, max_depth=2,random_state=master_seed, class_weight='balanced')
        estimator_train = estimator
        estimator_test = None
    elif estimator_name == 'MonotoneCLRF':
        estimator = CalibratedClassifierCV(RandomForestClassifier(n_estimators=500, max_depth=2,random_state=master_seed))
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None
    elif estimator_name == 'MonotoneRF':
        estimator = RandomForestClassifier(n_estimators=500, max_depth=2,random_state=master_seed)
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None
    elif estimator_name == 'XGb':
        estimator = XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, use_label_encoder=False, random_state=master_seed)
        estimator_train = estimator
        estimator_test = None            
    elif estimator_name == 'CLXGb':
        estimator = CalibratedClassifierCV(XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, use_label_encoder=False, random_state=master_seed))
        estimator_train = estimator
        estimator_test = None
    elif estimator_name == 'MonotoneCLXGb':
        estimator = CalibratedClassifierCV(XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, use_label_encoder=False, random_state=master_seed))
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None
    elif estimator_name == 'MonotoneXGb':
        estimator = XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, use_label_encoder=False, random_state=master_seed)
        estimator_train = FrankAndHallMonotoneClassifier(estimator)   
        estimator_test = None                
    else:
        raise ValueError('Unknwon estimator')

    print('Fitting Training Estimator')
    if model == '':
        estimator_train.fit(X_train, y_train)
        print('Saving Training Estimator')
        joblib.dump(estimator_train, estimator_name + '.pkl') 
        print('Training Estimator Saved')

    print('Training Estimator fitted', flush=True)
    probs_train = estimator_train.predict_proba(X_train)

    #  predictions_train = None
    print('Prediction_train computed')
    # CC
    cc = CC()
    cc.fit(X_train, y_train, predictions_train=probs_train)
    print('CC fitted')
    # PCC
    pcc = PCC()
    pcc.fit(X_train, y_train, predictions_train=probs_train)
    print('PCC fitted')
    # AC
    ac = AC(distance='L2')
    ac.fit(X_train, y_train, predictions_train=probs_train)
    print('AC fitted')
    # PAC
    pac = PAC(distance='L2')
    pac.fit(X_train, y_train, predictions_train=probs_train)
    print('PAC fitted')
    # EM
    em = EM()
    em.fit(X_train, y_train, predictions_train=probs_train)
    print('EM fitted')
    #  HDY
    hdy4 = HDy(n_bins=4, bin_strategy='equal_width')
    hdy4.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy fitted')
    #  HDY
    hdy8 = HDy(n_bins=8, bin_strategy='equal_width')
    hdy8.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy fitted')
    #  HDY
    hdy16 = HDy(n_bins=16, bin_strategy='equal_width')
    hdy16.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy fitted')
    #  CDFy-L1
    cdfy_l1 = DFy(distribution_function='CDF', n_bins=100, distance='L1', bin_strategy='equal_count')
    cdfy_l1.fit(X_train, y_train, predictions_train=probs_train)
    print('CDF-L1 fitted')
    #  EDy
    edy_eu = EDy(distance=euclidean_distances)
    edy_eu.fit(X_train, y_train, predictions_train=probs_train)
    print('EDy EU fitted')
    edy_emd = EDy(distance=emd_distances)
    edy_emd.fit(X_train, y_train, predictions_train=probs_train)
    print('EDy EMD fitted')
    #  PDFOrdinalY
    pdfy8_l2 = PDFOrdinaly(n_bins=8, distance='L2')
    pdfy8_l2.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy8 L2 fitted')
    pdfy16_l2 = PDFOrdinaly(n_bins=16, distance='L2')
    pdfy16_l2.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy16 L2 fitted')
    pdfy32_l2 = PDFOrdinaly(n_bins=32, distance='L2')
    pdfy32_l2.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy32 L2 fitted')
    pdfy8_emd = PDFOrdinaly(n_bins=8, distance='EMD')
    pdfy8_emd.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy8 EMD fitted')
    pdfy16_emd = PDFOrdinaly(n_bins=16, distance='EMD')
    pdfy16_emd.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy16 EMD fitted')
    pdfy32_emd = PDFOrdinaly(n_bins=32, distance='EMD')
    pdfy32_emd.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy32 EMD fitted')
    
    print('Fitting Estimator Test')
    if estimator_test is None:
        estimator_test = estimator_train
    else:
        estimator_test.fit(X_train, y_train)
    print('Estimator test fitted')

    prev_preds =  np.zeros((len(method_name), n_bags, 5))
    results_macro_nmd = np.zeros((len(method_name),1))
    results_nmd = np.zeros((len(method_name),1))
    for n_bag in range(n_bags):
        print('Validation Bag #%d' % n_bag, flush=True)
        X_test = load_testing_bag('../datasets/dev_samples/' + str(bag_inicial + n_bag) + '.txt')

        probs_test = estimator_test.predict_proba(X_test)
        prev_preds[0, n_bag, :] = cc.predict(X=None, predictions_test=probs_test).squeeze()
        prev_preds[1, n_bag, :] = pcc.predict(X=None, predictions_test=probs_test).squeeze()
        prev_preds[2, n_bag, :] = ac.predict(X=None, predictions_test=probs_test).squeeze()
        prev_preds[3, n_bag, :] = pac.predict(X=None, predictions_test=probs_test)
        prev_preds[4, n_bag, :] = em.predict(X=None, predictions_test=probs_test)
        prev_preds[5, n_bag, :] = hdy4.predict(X=None, predictions_test=probs_test)
        prev_preds[6, n_bag, :] = hdy8.predict(X=None, predictions_test=probs_test)
        prev_preds[7, n_bag, :] = hdy16.predict(X=None, predictions_test=probs_test)
        prev_preds[8, n_bag, :] = cdfy_l1.predict(X=None, predictions_test=probs_test)
        prev_preds[9, n_bag, :] = edy_eu.predict(X=None, predictions_test=probs_test)
        prev_preds[10, n_bag, :] = edy_emd.predict(X=None, predictions_test=probs_test)
        prev_preds[11, n_bag, :] = pdfy8_l2.predict(X=None, predictions_test=probs_test)
        prev_preds[12, n_bag, :] = pdfy16_l2.predict(X=None, predictions_test=probs_test)
        prev_preds[13, n_bag, :] = pdfy32_l2.predict(X=None, predictions_test=probs_test)
        prev_preds[14, n_bag, :] = pdfy8_emd.predict(X=None, predictions_test=probs_test)
        prev_preds[15, n_bag, :] = pdfy16_emd.predict(X=None, predictions_test=probs_test)
        prev_preds[16, n_bag, :] = pdfy32_emd.predict(X=None, predictions_test=probs_test)
        
    prev_preds[ prev_preds < 0 ] = 0 
    prev_preds[ prev_preds > 1 ] = 1 

    print('Saving and Computing results')    
    #  printing and saving results
    filename = 'results/Validation-T3-' + estimator_name  
    file_avg = open(filename + str(n_bags) + '.txt', 'w')

    n_classes = 5

    prev_true = ResultSubmission.load(file_preds)
    prev_true2 = load_prevalences(file_preds)

    for n_method, method in enumerate(method_name):    
        name_file_preds = 'predictions/preds-' + method + '-' + estimator_name + '.txt'
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

        macro_nmd = evaluate_submission(prev_true, prev_preds_method, n_bags, measure='macro-nmd', average=False)
        mnmd = evaluate_submission(prev_true, prev_preds_method, n_bags, measure='nmd', average=False)
 
        file_avg.write('%-15s' % (method))

        file_avg.write('%.5f (%.5f) '   %(macro_nmd.mean(), macro_nmd.std()))
        file_avg.write('%.5f (%.5f)\n'  %(mnmd.mean(), mnmd.std()))
 
        print('%-15s' % (method))

        print('%.5f (%.5f) '   %(macro_nmd.mean(), macro_nmd.std()))
        print('%.5f (%.5f)\n'  %(mnmd.mean(), mnmd.std()))
 

if __name__ == '__main__':
    main(estimator_name='MonotoneXGb', model='', file_preds='../datasets/dev_prevalences.txt', n_bags=1000, bag_inicial=0, master_seed=2032)
