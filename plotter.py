import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_conventional_vs_fh_models(sep:int, rep:int, estimator:str, values:list[int]) -> None:
    path = r"data\artificial\avg-artificial-ordinal-" + estimator + "-sep-" + str(sep) + "-rep" + str(rep) + ".csv"
    df = pd.read_csv(path)

    for col in df.columns:
        if col[0:2] != 'FH':
            plt.plot(values, df[col], label=col)
            plt.plot(values, df["FH-" + col], label="FH-" + col)
            plt.legend(loc="right")
            plt.ylabel("EMD")
            plt.savefig(r'data\figures\EMD_' + estimator + "-sep-" + str(sep) + '-' + col, bbox_inches='tight')


def plot_pred_vs_real_prevs(methods_names:list[str], values:int, n_classes:int, sep:int, bags:int, estimator:str) -> None:

    for method in methods_names:
        path_true = r'data\artificial_preds\true-preds-' + str(sep) + '-bags-' + str(bags)+ '.txt'
        path_pred = r'data\artificial_preds\pred-' + estimator + '-sep-' + str(sep) + '-' + method + '-bags-' + str(values) + '.txt'
        path_pred_FH = r'data\artificial_preds\pred-' + estimator + '-sep-' + str(sep) + '-FH-' + method + '-bags-' + str(values) + '.txt'

        df_true = pd.read_csv(path_true)
        df_pred = pd.read_csv(path_pred)
        df_pred_FH = pd.read_csv(path_pred_FH)


        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(n_classes):
            ax.scatter(df_true[str(c)], df_pred[str(c)], label=f'Clase {str(c+1)}')

        ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
        ax.set_xlabel('Prevalencia Real')
        ax.set_ylabel('Prevalencia Predicha')
        ax.set_title(f'Prevalencia Real vs. Predicción en {method}', fontsize=20)
        ax.legend(prop={'size': 17})
        plt.savefig(r'data\figures\pred_vs_real_' + method, bbox_inches='tight')


        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for c in range(n_classes):
            ax2.scatter(df_true[str(c)], df_pred_FH[str(c)], label=f'Clase {str(c+1)}')

        ax2.plot([0, 1], [0, 1], 'k--', label='Ideal')
        ax2.set_xlabel('Prevalencia Real')
        ax2.set_ylabel('Prevalencia Predicha')
        ax2.set_title(f'Prevalencia Real vs. Predicción en FH-{method}', fontsize=20)
        ax2.legend(prop={'size': 17})
        plt.savefig(r'data\figures\pred_vs_real_FH-' + method, bbox_inches='tight')


    
if __name__ == '__main__':
    seps = [3, 6]
    estimator = "LR"
    rep = 10
    values = [50, 100, 500, 1000, 1500, 2000]
    methods_names = ['AC', 'PAC', 'EM', 'EDy', 'HDy']
    values_preds = 500
    n_classes = 5
    bags = 500
    for sep in seps:
        plot_conventional_vs_fh_models(sep, rep, estimator, values)
        plot_pred_vs_real_prevs(methods_names, values_preds, n_classes, sep, bags, estimator)

