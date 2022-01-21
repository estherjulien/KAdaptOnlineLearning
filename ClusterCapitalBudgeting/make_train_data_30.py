from main_learn_suc_pred import main
import numpy as np
import sys

if __name__ == "__main__":
    array_task = int(sys.argv[1])
    N_train = int(sys.argv[2])
    K_train = int(sys.argv[3])
    for i in np.arange((array_task - 1)*30, array_task*30):
        main(i, N_train, K_train)
