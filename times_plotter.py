import matplotlib.pyplot as plt
import pickle
import numpy as np

times = {}

with open("times.dat", "rb") as f:
    times = pickle.load(f)

    # Prepare the data
    x = np.array(range(len(times["calculateMatrix"])))

    plt.yscale("log")
    plt.title("Time taken to produce 10 million element buffer")
    plt.ylabel("Time taken (ns) [log scale]")
    plt.xlabel("Trial number")

    for fn in times:
        plt.scatter(x,times[fn], label=fn)

        print("Mean time for {}: {} ns".format(fn, sum(times[fn])/len(times[fn])))

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()