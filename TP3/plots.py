import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import seaborn as sns


def read_times(logs_folder):
    files = listdir(logs_folder)
    times_specs = []

    for fn in files:
        # read file, the time is recorded at the line -1
        if fn.startswith('logs'):
            with open(logs_folder+fn, 'r', encoding="utf8") as f:
                lines = f.readlines()
                specs = fn.split('-')
                if "Total Time:" in lines[-2]:
                    print(fn)
                    time = float(lines[-1])
                    print(f"Time: {time:.2f} s")
                    times_specs.append((specs[2][2:],specs[3][2:],time))
                else:
                    print(f"File {fn} is incomplete")
            f.close()
    return times_specs

def read_evals(logs_folder): # Time or score
    files = listdir(logs_folder)
    evals_specs = []
    for fn in files:
        # read file, the time is recorded at the line -1
        if fn.startswith('logs'):
            with open(logs_folder+fn, 'r', encoding="utf8") as f:
                lines = f.readlines()
                specs = fn.split('-')
                if "score" in lines[-2]:
                    print(fn)
                    score = float(lines[-1])
                    print(f"Score: {score:.2f} %")
                    evaluation_metric = "by_order" if "by_order" in lines[2] else "no_order"
                    evals_specs.append((specs[3],specs[4],score,evaluation_metric))
                else:
                    print(f"File {fn} is incomplete")
            f.close()
    return evals_specs

def plot_eval_per_time(df1,df2):

    fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(10,4))

    #ax.legend()
    #ax.grid(True)
    title = f"Performance (metric='{df1.name}' or '{df2.name}' ) / Temps d'exécution"
    fig.suptitle(title)
    # Customize the major grid
    ax1.grid(which='major', linestyle='--', linewidth='0.5', color='black')
    ax2.grid(which='major', linestyle='--', linewidth='0.5', color='black')
    # Customize the minor grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    ax1.set_ylim([0,90])
    ax2.set_ylim([0,90])
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlabel("Temps de correction (en sec) - échelle logarithmique")
    ax1.set_ylabel(f"La performance du correcteur '{df1.name}' (en %)")
    ax2.set_xlabel("Temps de correction (en sec) - échelle logarithmique")
    ax2.set_ylabel(f"La performance du correcteur '{df2.name}' (en %)")
    fig.tight_layout()
    sns.scatterplot(ax=ax1,data=df1, x="Temps", y="Performance", hue="Distance", style="Ordre",
                    sizes=[0.9 for n in range(len(df1))],
                    markers=['^','X','P', '>'], legend=False)
    sns.scatterplot(ax=ax2,data=df2, x="Temps", y="Performance", hue="Distance", style="Ordre",
                    sizes=[0.9 for n in range(len(df1))],
                    markers=['^','X','P', '>'], legend="full")

    plt.subplots_adjust(right=0.78)

    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figtitle = f"out/eval-Performance_{df1.name}_{df2.name}-Temps"
    fig.savefig(f"{figtitle}.svg",format="svg")
    fig.savefig(f"{figtitle}.eps",format="eps")
    fig.savefig(f"{figtitle}.pdf")
    plt.show()

def main():
    times_specs = read_times('logs/')
    evals_specs = read_evals('eval/')

    d1 = pd.DataFrame(times_specs, columns = ['Distance', 'Ordre', 'Temps'])
    d2 = pd.DataFrame(evals_specs, columns = ['Distance', 'Ordre', 'Performance', "Metric"])
    resultat = pd.merge(d1, d2,  on=['Distance','Ordre'])
    groups = resultat.groupby("Metric")
    no_order = groups.get_group('no_order')
    no_order.name = 'no_order'
    by_order = groups.get_group('by_order')
    by_order.name='by_order'
    resultat.to_csv('resultat.csv')
    no_order.to_csv('no_order.csv')
    by_order.to_csv('by_order.csv')
    plot_eval_per_time(no_order,by_order)




if __name__ == '__main__':
    main()