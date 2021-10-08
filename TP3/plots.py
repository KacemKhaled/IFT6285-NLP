import matplotlib.pyplot as plt
from os import listdir
import pandas as pd


def read_times(logs_folder):
    files = listdir(logs_folder)
    times_specs = []

    for fn in files:
        # read file, the time is recorded at the line -1
        if fn.startswith('logs'):
            with open(logs_folder+fn, 'r', encoding="utf8") as f:
                lines = f.readlines()
                specs = fn.split('-')
                print(specs)
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
                print(specs)
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

def plot_eval_per_time(times,evals, corrections):
    fig, ax = plt.subplots(figsize=(8,7))
    for i in range(len(corrections)):

        ax.scatter(times[i],evals[i],  label=corrections[i],
                   alpha=1)

    ax.legend()
    ax.grid(True)
    plt.title("Performance/Temps d'exécution")
    plt.xlabel("Temps mis dans la correction (en secondes)")
    plt.ylabel("La performance du correcteur")
    fig.savefig("out/eval-time-"+str(len(corrections)-1)+" transformations.svg",format="svg")
    plt.show()

def main():
    times_specs = read_times('logs/')
    evals_specs = read_evals('eval/')

    d1 = pd.DataFrame(times_specs, columns = ['Distance', 'Ordre', 'Temps'])
    d2 = pd.DataFrame(evals_specs, columns = ['Distance', 'Ordre', 'Performance', "Metric"])
    resultat = pd.merge(d1, d2,  on=['Distance','Ordre'])
    print(resultat)



if __name__ == '__main__':
    main()