import numpy as np
import matplotlib.pyplot as plt

def read_and_plot():
    files = ["hitpointx_C1.csv", "hitpointx_C2.csv","hitpointx_C3.csv", "hitpointx_C4.csv"]
    legend_txt = ["cushion 1", "cushion 2","cushion 3", "cushion 4"]
    
    data_arrays = []
    for f in files:
        data = np.loadtxt(f, delimiter=',')
        data_arrays.append(data)
    NN = len(data_arrays[0])
    phi_range = np.linspace(60, 61.01, NN)
    for i, arr in enumerate(data_arrays):
        localNN = len(arr)
        local_phi_range = np.linspace(60, 61.01, localNN)
        plt.plot(local_phi_range, arr-arr.min(), label=legend_txt[i])
    plt.xlabel("Phi")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    read_and_plot()