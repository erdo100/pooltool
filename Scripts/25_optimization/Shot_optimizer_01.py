from tkinter import Tk
from parameters import Parameters
from billiardenv import BilliardEnv
from GUI import plot_3cushion
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

   
if __name__ == '__main__':
    # Initialize components
    params = Parameters()
    sim_env = BilliardEnv()
    
    # Create and run GUI
    plt3c = plot_3cushion(sim_env, params)
    plt3c.root.mainloop()