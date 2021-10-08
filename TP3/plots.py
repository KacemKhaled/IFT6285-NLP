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

def plot_eval_per_time(df):
    fig, ax = plt.subplots(figsize=(8,7))

    #ax.legend()
    #ax.grid(True)
    title = f"Performance {str(df.name)} / Temps d'exécution"
    plt.title(title)
    plt.xlabel("Temps mis dans la correction (en secondes)")
    plt.ylabel("La performance du correcteur")
    #ax = df.plot.scatter(x="a", y="b", color="DarkBlue", label="Group 1")

    #df.plot.scatter(x="c", y="d", color="DarkGreen", label="Group 2", ax=ax)
    #df.plot.scatter(x="Temps", y="Performance", c="Distance", cmap="viridis", s=50)
    sns.scatterplot(data=df, x="Temps", y="Performance", hue="Distance", style="Ordre")

    fig.savefig(f"out/eval-Performance {df.name} - Temps d'exécution.svg",format="svg")
    fig.savefig(f"out/eval-Performance {df.name} - Temps d'exécution.eps",format="eps")
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
    plot_eval_per_time(no_order)
    plot_eval_per_time(by_order)



if __name__ == '__main__':
    main()