###############################################################################################
from datetime import datetime
import sys, os
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger()
###############################################################################################
def create_dirs(paths_list):
    for path in paths_list:
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
            exit(1)

##############################################################################################
def add_date_to_paths(config):
    """
    adds time and date signature to directories names
    """
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    config.problems_dir += dt_string
    config.subproblems_dir += dt_string
    config.plans_dir += dt_string
    config.img_dir += dt_string
    config.tables_dir += dt_string
    config.csv_path += dt_string + ".csv"

###############################################################################################
def save_info_df_as_csv(info_df, path):
    info_df.to_csv(path)

###############################################################################################
def change_prefix_ubu_2_windows(csv_path):
    windows_pre = r"C:/Users/NikiDavarashvili/OneDrive - Technion/Desktop/Project/"
    ubunto_pre = r"/media/sf_Project/"
    df = pd.read_csv(csv_path)

    def change_pre(path):
        after_pre = path.partition(ubunto_pre)[-1]
        if after_pre == '':
            logger.debug("Bad Name!!!")
            return path

        return os.path.join(windows_pre, after_pre)

    for column in ['problem', 'plan', 'table']:
        df[column] = df[column].apply(change_pre)

    save_info_df_as_csv(df, csv_path)


###############################################################################################
def change_prefix_windows_2_ubu(csv_path):
    windows_pre = r"C:/Users/NikiDavarashvili/OneDrive - Technion/Desktop/Project/"
    ubunto_pre = r"/media/sf_Project/"
    df = pd.read_csv(csv_path)

    def change_pre(path):
        after_pre = path.partition(windows_pre)[-1]
        if after_pre == '':
            logger.debug("Bad Name!!!")
            return path

        return os.path.join(ubunto_pre, after_pre)

    for column in ['problem', 'plan']:
        df[column] = df[column].apply(change_pre)

    save_info_df_as_csv(df, csv_path)

def find_test_train_duplicates(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    def check_same_table(train_row, table2_path):
        arr1 = np.load(train_row['table'])
        arr2 = np.load(table2_path)
        res = np.array_equiv(arr1, arr2)
        return res

    same_list = []
    for idx, row in test_df.iterrows():
        eq_length_train_df = train_df[train_df['plan length'] == row['plan length']]
        print("current shape: {}".format(eq_length_train_df.shape))
        compare_func = lambda x: check_same_table(x, row['table'])
        eq_length_train_df = eq_length_train_df[eq_length_train_df.apply(compare_func, axis=1)]
        print("shape after: {}".format(eq_length_train_df.shape))

        if eq_length_train_df.shape[0] != 0:
            same_list.append(idx)

    print(same_list)






