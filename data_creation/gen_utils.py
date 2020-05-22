###############################################################################################
from datetime import datetime
import sys, os

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
def save_info_df_as_csv(info_df, config):
    info_df.to_csv(config.csv_path)