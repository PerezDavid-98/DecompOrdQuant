import numpy as np
import seaborn as sns
import pandas as pd

from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.df import HDy
from quantificationlib.multiclass.energy import EDy

from quantificationlib.metrics.ordinal import emd

from quantificationlib.estimators.frank_and_hall import FrankAndHallMonotoneClassifier
from quantificationlib.decomposition.ordinal import FrankAndHallQuantifier

from quantificationlib.bag_generator import PriorShift_BagGenerator

from sklearn.linear_model import LogisticRegression

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer


# Añadido para eliminar los warning del solver del estimador.
import warnings
warnings.filterwarnings("ignore")


# MAIN
n_classes = 5
nbags = 500
mu_sep = 3  # 3
sigma = 1.5

seed = 42
rng = np.random.RandomState(seed)

estimator_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
estimator = LogisticRegression(random_state=seed, max_iter=1000, solver="liblinear")

values = [1000]

nreps = 10

#   methods
methods_names = ['AC', 'PAC', 'EM', 'EDy', 'HDy', 'FH-AC', 'FH-PAC', 'FH-EM', 'FH-EDy', 'FH-HDy']

est_name = 'LR'

# to store all the results
results = np.zeros((len(methods_names), len(values)))


# Modifico una de las paletas de colores de seaborn para generar el grafico que muestra los datos generados (Desactivado durante pruebas)
# Get the default 'Blues' palette
blues_palette = sns.color_palette('Blues', n_colors=5)
custom_palette = [blues_palette[1], blues_palette[2], blues_palette[3], blues_palette[4], (0.1, 0.2, 0.6)]  # Shift and modify the lightest color



for k in range(len(values)):

    print()
    print('Train examples: ', values[k], end='')
    n_train = values[k]
    n_test = 2000  # n_train


    for rep in range(nreps):

        print()
        print('Rep#', rep + 1, end=' ', flush=True)
        # create datasets
        X_train = np.array([]).reshape(0,2)
        y_train = []
        y_train_binary = []
        X_test = np.array([]).reshape(0,2)
        y_test = []


        for i in range(n_classes):
            mu = mu_sep * (i + 1)

            training_examples = sigma * rng.randn(n_train, 2) + mu

            X_train = np.vstack((X_train, training_examples))
            y_train = np.append(y_train, (i + 1) * np.ones(n_train))
            if i < n_classes / 2:
                y_train_binary = np.append(y_train_binary, -np.ones(n_train))
            else:
                y_train_binary = np.append(y_train_binary, np.ones(n_train))

            testing_examples = sigma * rng.randn(n_test, 2) + mu

            X_test = np.vstack((X_test, testing_examples))
            y_test = np.append(y_test, (i + 1) * np.ones(n_test))


        # Se usa un DataFrame para plotear los datos (Desactivado durante pruebas)
        df= pd.DataFrame(X_train, columns = ['X1','X2'])
        df['Clase'] = y_train

        # Se genera el plot que muestra los datos generados (Desactivado durante pruebas)
        g=sns.pairplot(vars=["X1","X2"], data=df, hue="Clase", palette=custom_palette)
        g.figure.suptitle("Dataset Artificial - " + str(n_train), y=1.08)
        g.savefig('data\Figuras\\artificial_pairplot_' + est_name + '-sep-' + str(mu_sep) + '-values-' + str(n_train) + '.png')

        # fitting classifiers here, all methods use exactly the same predictions
        skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                          scoring=make_scorer(geometric_mean_score), n_jobs=None)
        gs.fit(X_train, y_train_binary)
        best_lr = gs.best_estimator_

        print(best_lr)
        
        estimator_fh = FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=None)

        #   create objects
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

        for nmethod, method in enumerate(methods):
            print(f"Fitting method: {methods_names[nmethod]}")
            method.fit(X=X_train, y=y_train)

        print('Test', end=' ')
        #  Testing bags
        bag_generator = PriorShift_BagGenerator(n_bags=nbags, bag_size=len(X_test),
                                                min_prevalence=None, random_state=seed + rep + 10)

        prev_true, indexes = bag_generator.generate_bags(X_test, y_test)
        
        name_true_pred_file = "true-preds-" + str(mu_sep) + '-bags-' + str(nbags) + ".txt"
        true_pred_file = open("data\\artificial_preds\\" + name_true_pred_file, 'w')
        true_pred_file.write('id')
        for i in range(n_classes):
            true_pred_file.write(',%d' % (i))
        true_pred_file.write('\n')
            
        for n_bag in range(nbags):
            true_pred_file.write('%d,' % (n_bag))
            prev_true[:, n_bag].tofile(true_pred_file, sep=',')
            true_pred_file.write('\n')
        true_pred_file.close()  


        prev_preds =  np.zeros((len(methods_names), nbags, 5))
        for n_bag in range(nbags):
            print("Predicting bag #%i" % n_bag)
            for index, method in enumerate(methods):
                prev_preds[index, n_bag, :] = method.predict(X=X_test[indexes[:, n_bag], :])

        prev_preds[ prev_preds < 0 ] = 0
        prev_preds[ prev_preds > 1 ] = 1

        print('Saving and Computing results')    

        for n_method, method in enumerate(methods_names):    
            name_pred_file = "pred-" + est_name + "-sep-" + str(mu_sep) + '-' + method + '-bags-' + str(nbags) + ".txt"
            pred_file = open("data\\artificial_preds\\" + name_pred_file, 'w')
            pred_file.write('id')
            for i in range(n_classes):
                pred_file.write(',%d' % (i))
            pred_file.write('\n')
            
            for n_bag in range(nbags):
                pred_file.write('%d,' % (n_bag))
                prev_preds[n_method, n_bag, :].tofile(pred_file, sep=',')
                pred_file.write('\n')
            pred_file.close()
        
        
        for n_bag in range(nbags):
            for n_method, _ in enumerate(methods_names):
                error = emd(prev_true[:, n_bag], prev_preds[n_method, n_bag, :])
                results[n_method, k] = results[n_method, k] + error


results = results / (nreps * nbags)

name_file = "avg-artificial-ordinal-" + est_name + "-sep-" + str(mu_sep) + "-rep" + str(nreps) + ".txt"
file_avg = open("data\\artificial\\" + name_file, 'w')

for index, m in enumerate(methods_names):
    file_avg.write('%s,' % m)
file_avg.write('\n')


for i in range(len(values)):
    for index, _ in enumerate(methods_names):
        file_avg.write('%.5f,' %  results[index, i])
    file_avg.write('\n')

file_avg.close()


print('', flush=True)