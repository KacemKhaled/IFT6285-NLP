import matplotlib.pyplot as plt
from os import listdir


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

def read_evals(logs_folder, keyword="score",pos1=3,pos2=4 ): # Time or score
    files = listdir(logs_folder)
    evals_specs = []
    for fn in files:
        # read file, the time is recorded at the line -1
        if fn.startswith('logs'):
            with open(logs_folder+fn, 'r', encoding="utf8") as f:
                lines = f.readlines()
                specs = fn.split('-')
                print(specs)
                if keyword in lines[-2]:
                    print(fn)
                    score = float(lines[-1])
                    print(f"{keyword}: {score:.2f} %")
                    evals_specs.append((specs[pos1],specs[pos2],score))
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
    print(times_specs)
    print(t[1:2])
    times = []
    evals = []
    if len(times_specs)==len(evals_specs):
        for t in times_specs:
            if t[1:2] in evals_specs:
                plot_eval_per_time([t[2] for t in times_specs],[e[2] for e in evals_specs],t[1:2])

if __name__ == '__main__':
    main()