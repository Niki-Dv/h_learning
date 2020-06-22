# imports
import os, sys
import logging
import subprocess
import pandas as pd
import re

# add package path to sys
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.join(curr_dir_path, '..')
if package_path not in sys.path:
    sys.path.append(package_path)

from data_creation import h_config

config = h_config.RoversConfig().get_config()

logger = logging.getLogger()

csv_path = r"/media/sf_Project/Data_generator/generated_problems/goal_as_column/csv_dir/info_24_05_2020_22_45_07.csv"
save_csv_path = r"/media/sf_Project/Data_generator/generated_problems/goal_as_column/csv_dir/HEUR_info_24_05_2020_22_45_07.csv"
stdout_path = r"/media/sf_Project/Data_generator/temp/niki.txt"

def get_heurstic_result(heurstic_name, df):
    planner_search_flag = "--search \"astar(" + heurstic_name + ")\""
    df[planner_search_flag] = -1
    for i in range(0, df.shape[0]):

        cmd_line_args = ["exec", "python", config.planner_path, config.domain_pddl_path,
                         df.at[i, "problem"], planner_search_flag]
        cmd_line = " ".join(cmd_line_args)
        with open(stdout_path, 'w') as out_f:
            try:
                sub_proc = subprocess.Popen(cmd_line, stdout=out_f, stderr=None, shell=True)
                sub_proc.wait(timeout=config.plan_finding_timeout)
            except:
                sub_proc.kill()
                logger.warning("reached to time out for finding plan for problem {} , continuing".format(i))
                continue
        with open(stdout_path, 'r') as out_f:
            for line in out_f:
                x = re.search("Initial heuristic value[^1-9]+([1-9]+)", line)
                if x is not None:
                    df.at[i, planner_search_flag] = int(x[1])

        logger.debug("Found plan for problem {}".format(i))


    return df

if __name__ == '__main__':
    logger.info("Started creating single objective script")
    res_df = pd.read_csv(csv_path)
    res_df = get_heurstic_result('ff', res_df)
    res_df.to_csv(save_csv_path)
