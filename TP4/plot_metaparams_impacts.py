import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import numpy as np
import seaborn as sns


def sns_line_plot(ax,d,variable,fix1,value1,fix2,value2):
    data = d.groupby(fix1).get_group(value1).groupby(fix2).get_group(value2)
    sns.lineplot(ax=ax,data=data, 
    x='Sentences', y='Times', hue=variable, sizes=0.9, legend=True) 
    ax.set_title(f"Variable: {variable}\nFixes: '{value1}' et '{value2}'", fontsize=12)
    
    ax.set_xlabel(f"Nombre de phrases traitées")
    ax.set_ylabel(f"Le temps d'entrainement (s)")

def plot_1_3(folder,csv_files):
    
    fig, (ax1,ax2,ax3)= plt.subplots(1, 3, figsize=(10,3))
    
    title = f"Influence des meta-paramètre sur les temps d'entrainenemt (s) en fonction de phrases traitées"
    #fig.suptitle(title)
    values =[]

    for f in csv_files:
        times,sizes,sent_len = np.loadtxt(folder+f,delimiter=',')

        sent_len  = [int(l) for l in sent_len]
        label = f[7:-14]
        s_size = f[f.find('size'):f.find('window')-1]
        s_window = f[f.find('window'):f.find('neg')-1]
        s_negative = f[f.find('neg'):f.find('mincount')-1]
        for i in range(len(sent_len)):
            values.append((times[i],sent_len[i],s_size,s_window,s_negative))
        d = pd.DataFrame(values, columns = ['Times', 'Sentences',  'Vector Size','Window','Negative'])
    # print(d)
    
    sns_line_plot(ax=ax1,d=d,variable='Vector Size',fix1='Window',value1='window5',fix2='Negative',value2='neg5')
    sns_line_plot(ax=ax2,d=d,variable='Window',fix1='Vector Size',value1='size100',fix2='Negative',value2='neg5')
    sns_line_plot(ax=ax3,d=d,variable='Negative',fix1='Window',value1='window5',fix2='Vector Size',value2='size100')
    plt.legend()
    fig.tight_layout() 
    figname= 'times-sentences-metaparams'
    d.to_csv(f'outputs/{figname}.csv',  index = False)
    plt.savefig(f"plots/{figname}.svg",format="svg")
    plt.savefig(f"plots/{figname}.png", format="png")
    plt.savefig(f"plots/{figname}.eps", format="eps")
    #plt.show()

def generate_df(folder,csv_files):
    
    fig, (ax1,ax2,ax3)= plt.subplots(1, 3, figsize=(10,3))
    
    values =[]
    for f in csv_files:
        times,sizes,sent_len = np.loadtxt(folder+f,delimiter=',')

        sent_len  = [int(l) for l in sent_len]
        label = f[7:-14]
        s_size = f[f.find('size'):f.find('window')-1]
        s_window = f[f.find('window'):f.find('neg')-1]
        s_negative = f[f.find('neg'):f.find('mincount')-1]
        #for i in range(len(sent_len)):
        values.append((times,sizes,sent_len,s_size,s_window,s_negative))
        d = pd.DataFrame(values, columns = ['Times','Sizes', 'Sentences',  'Vector Size','Window','Negative'])
    # print(d)
    
    figname= 'times-sizes-sentences-metaparams'
    d.to_csv(f'outputs/{figname}.csv',  index = False)

def main():
    folder ='outputs/'
    models= 'models/'
    files = listdir(folder)
    csv_files = [f for f in files if  f.startswith('genw2v') and f.endswith('.csv')]
    plot_1_3(folder,csv_files)
    generate_df(folder,csv_files)

if __name__ == '__main__':
    main()