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
# Shot 4:
# Loss: 0.1732663642 - Best Params: [-0.30788, 0.33058, 102.28146, 5.02267, 0.04398, 0.01771, 0.19830, 0.90065, 0.84528, 0.22216, 0.03477]