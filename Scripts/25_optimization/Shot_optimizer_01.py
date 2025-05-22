from tkinter import Tk
from parameters import Parameters
from billiardenv import BilliardEnv
from GUI import plot_3cushion

   
if __name__ == '__main__':
    # Initialize components
    params = Parameters()
    sim_env = BilliardEnv()
    
    # Create and run GUI
    plt3c = plot_3cushion(sim_env, params)
    plt3c.root.mainloop()


    # Best: [0.14170, -0.06764, -85.81472, 4.73570, 4.01064, 0.18762, 0.46916, 0.67855, 0.29084, 0.00383, 0.01267, 0.85868, 0.74928, 0.40936, 0.03393]  Loss: 33.69313