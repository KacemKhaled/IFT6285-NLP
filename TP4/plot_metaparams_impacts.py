import matplotlib.pyplot as plt
from os import listdir,path
import pandas as pd
import numpy as np

import seaborn as sns


def plot_size_per_time(df1):

    fig, ax1= plt.subplots(1, 1, figsize=(6,4))

    #ax.legend()
    #ax.grid(True)
    title = f"Taille (MB)) / Temps d'entrainenemt (s)"
    fig.suptitle(title)
    # Customize the major grid
    ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
    # Customize the minor grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    #ax1.set_ylim([0,90])
    #ax1.set_xlim([100,10000])
    ax1.set_xlabel("Temps d'entrainenemt (en sec)")
    ax1.set_ylabel(f"La taille du modèle (en MB)")
    fig.tight_layout() # 'Time', 'Size',  'Vector Size','Window','Negative'
    sns.scatterplot(ax=ax1,data=df1, x="Time", y='Vector Size', hue='Window', style='Negative',sizes=df1['Size'], legend=True)
    plt.subplots_adjust(right=0.8)

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figtitle = f"plots/Taille-Temps"
    fig.savefig(f"{figtitle}.svg",format="svg")
    fig.savefig(f"{figtitle}.eps",format="eps")
    fig.savefig(f"{figtitle}.pdf")
    plt.show()



def main():
    folder ='outputs/'
    models= 'models/'
    files = listdir(folder)
    values = []
    csv_files = [f for f in files if f.endswith('.csv')]
    plt.figure(figsize=(9, 6))
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(csv_files)))))

    for f in csv_files:
        times,sizes,sent_len = np.loadtxt(folder+f,delimiter=',')
        #print(f,times,sizes,sent_len,sep='\n',end='\n\n')
        real_final_size = (path.getsize(models+f[:-4]+'.w2v')+path.getsize(models+f[:-4]+'.w2v.syn1neg.npy')+path.getsize(models+f[:-4]+'.w2v.wv.vectors.npy') )/ (1024*1024) # convert to MB
        #print(real_final_size)
        plt.plot([int(l) for l in sent_len], times,label=f[7:-14])
        s_size = f[f.find('size'):f.find('window')-1]
        s_window = f[f.find('window'):f.find('neg')-1]
        s_negative = f[f.find('neg'):f.find('mincount')-1]
        values.append((times[-1],real_final_size,s_size,s_window,s_negative))
    plt.title(f"Le temps mis pour entrainer les modeles en fonction du nombre de phrases considérées / parametres")
    plt.xlabel("Nombre de phrases considérées")
    plt.ylabel("Le temps mis pour entrainer les modeles (en secondes)")
    plt.legend()
    plt.savefig(f"plots/courbe-hyper.svg",format="svg")
    plt.savefig(f"plots/courbe-hyper.png", format="png")
    plt.savefig(f"plots/courbe-hyper.eps", format="eps")
    plt.show()

    d = pd.DataFrame(values, columns = ['Time', 'Size',  'Vector Size','Window','Negative'])
    print(d)
    plot_size_per_time(d)



if __name__ == '__main__':
    main()