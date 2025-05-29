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

# [-0.19768, 0.15808, 113.78364, 2.70124, 5.44760, 0.03783, 0.23999, 0.00542, 0.32554, 0.97317, 0.85508, 0.23228]
#
# [-0.18600, 0.02035, 92.97000, 3.43000, 0.00000, 0.02701, 0.22564, 0.00646, 0.33000, 0.93421, 0.86104, 0.14416]