###############################################################################################
from datetime import datetime
import sys, os
import pandas as pd
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
    windowd_pre = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project"
    ubunto_pre = r"/media/sf_Project"
    df = pd.read_csv(csv_path)

    def change_pre(path):
        after_pre = path.partition(ubunto_pre)
        return os.path.join(windowd_pre, after_pre)

    for column in ['problem', 'plan', 'table', 'image']:
        df[column] = df[column].apply(change_pre)

    save_info_df_as_csv(df, csv_path)


