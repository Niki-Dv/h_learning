import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TEST_DATA_CSV_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\Final_data\Heurstic_final\heur_Rovers_2_27755.csv"
TRAIN_DATA_CSV_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\Final_data\Rovers_27755_balanced.csv"
timestr = time.strftime("%Y%m%d-%H%M%S")
PLOT_TEST_ERR_NAME = "Test err bar_" + timestr + "_.png"
PLOT_TRAIN_ERR_NAME = "Train err bar_" + timestr + "_.png"

PLOT_SAMPLES_NAME = "Number of samples per plan length_" + timestr+ "_.png"


PLOTS_SAVE_PATH = r'C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results\plots'
NET_COL = "Net Heur"
PLAN_LEN_COL = 'plan length'


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}



def collect_data(df):
    mean_list, var_list, num_samples_list = [], [], []
    possible_length = list(df[PLAN_LEN_COL].unique())
    possible_length.sort()
    for plan_length in possible_length:
        matching_df = df[df[PLAN_LEN_COL] == plan_length]
        diff_ser = matching_df[PLAN_LEN_COL] - matching_df[NET_COL]

        if len(diff_ser) < 5:
            continue
        mean_list.append(diff_ser.mean())
        var_list.append(diff_ser.std())
        num_samples_list.append(len(diff_ser))

        print(
            f"Mean value for length: {plan_length} is: {mean_list[-1]} and variance: {var_list[-1]}. (used {num_samples_list[-1]} sampels)")

    return possible_length[:len(mean_list)], mean_list, var_list, num_samples_list

def plot_err_bar(possible_length, mean_list, var_list, plot_name):
    plt.figure(figsize=[20, 12])
    plt.errorbar(possible_length[:len(mean_list)], mean_list, yerr=var_list, fmt='o', color='black',
                 ecolor='gray', elinewidth=3, capsize=0)

    plt.xlim(possible_length[0], possible_length[-1])
    plt.xticks(possible_length)
    plt.grid()
    plt.title("Error bar (mean,std) of the vector: Actual Length - Net Heuristic", **font)
    plt.xlabel("Actual Plan Length")
    plt.ylabel("Error (Mean, Std)")
    plt.savefig(os.path.join(PLOTS_SAVE_PATH, plot_name))


def main():
    train_df = pd.read_csv(TRAIN_DATA_CSV_PATH)
    test_df = pd.read_csv(TEST_DATA_CSV_PATH)

    tr_samples_length, tr_mean_list, tr_var_list, tr_num_samples_list = collect_data(train_df)
    test_samples_length, test_mean_list, test_var_list, test_num_samples_list = collect_data(test_df)


    # create figure of number of samples
    plt.figure(figsize=[18, 12])
    plt.scatter(tr_samples_length, tr_num_samples_list, color='red', label='train')
    plt.scatter(test_samples_length, test_num_samples_list, color='blue', label='test')

    plt.grid()
    plt.title("Number of samples per plan length", **font)
    plt.xlabel("Actual Plan Length")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH, PLOT_SAMPLES_NAME))

    plot_err_bar(tr_samples_length, tr_mean_list, tr_var_list, PLOT_TRAIN_ERR_NAME)
    plot_err_bar(test_samples_length, test_mean_list, test_var_list, PLOT_TEST_ERR_NAME)


if __name__ == '__main__':
    main()
