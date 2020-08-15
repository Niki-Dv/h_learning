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

csv_path = os.path.join(r"/media/sf_Project/Data_generator/Final_data/Hiking_13120_2.csv")
save_csv_path = os.path.join(r"/media/sf_Project/Data_generator/Final_data/Heurstic_final/Heur_Hiking_13120_2.csv")
stdout_path = os.path.join(r"/media/sf_Project/Data_generator/temp/niki.txt")

def get_heurstic_result(heurstics_names, df):
    for heuristic_name in heurstics_names:
        planner_search_flag = "--search \"astar(" + heuristic_name + ")\""
        df[planner_search_flag] = -1
        for i in range(0, df.shape[0]):
            print(f"at {i}/{df.shape[0]}")
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
                        print(f"actuall length: {df.loc[i]['plan length']}. heurstic: {heuristic_name} value: {int(x[1])}")
                        df.at[i, planner_search_flag] = int(x[1])
                        break

            logger.debug("Found plan for problem {}".format(i))
        #df = df[df[planner_search_flag] != -1]

    return df

if __name__ == '__main__':
    logger.info("Started creating single objective script")
    res_df = pd.read_csv(csv_path)
    heuristic_to_test = ['add', 'lmcount(lm_exhaust)', 'lmcut', 'cegar', 'ff']
    res_df = get_heurstic_result(heuristic_to_test, res_df)
    res_df.to_csv(save_csv_path)
