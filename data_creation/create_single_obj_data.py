# imports
import os, sys
import logging
import re

# add package path to sys
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.join(curr_dir_path, '..')
if package_path in sys.path:
    sys.path.append(package_path)

from data_creation import gen_utils, planning_utils, h_config, create_table

config = h_config.config().get_config()

logger = logging.getLogger()

PROBLEM_PATH_COLUMN = "problem"
THE_TRUE_GOAL = "(:goal (and\n\
(communicated_soil_data waypoint0)\n\
(communicated_image_data objective0 low_res)\n )\n) \n)"

##############################################################################################
def switch_to_same_objective(df):

    for idx, row in df.iterrows():
        problem_path = row[PROBLEM_PATH_COLUMN]
        with open(problem_path, "r") as f:
            lines = f.readlines()
        with open(problem_path, "w") as f:
            for line in lines:
                if re.search(":goal", line, re.IGNORECASE):
                    break
                f.write(line)
            f.write(THE_TRUE_GOAL)
        

##############################################################################################
if __name__ == '__main__':
    h_config.define_logger(logger, config.logger_path)
    logger.info("Started creating single objective script")
    gen_utils.add_date_to_paths(config)
    gen_utils.create_dirs([config.plans_dir, config.problems_dir, config.subproblems_dir, config.img_dir, config.tables_dir])
    logger.debug("Finished preparing directories and paths, starting generating data")

    NProbDF = planning_utils.GenProblemsParams(config)
    NProbDF = planning_utils.GenerateProblems(NProbDF)
    logger.debug("Finished creating problems")

    switch_to_same_objective(NProbDF)
    logger.debug("Finished switching targets")
    NProbDF = planning_utils.SolveProblems(NProbDF, config)

    NProbDF = planning_utils.delete_unsolved_problems(NProbDF)
    NProbDF.reset_index(drop=True, inplace=True)
    # save for case of interrupt
    gen_utils.save_info_df_as_csv(NProbDF, config.csv_path)
    logger.debug("Deleted unsolved problems")

    NProbDF = planning_utils.ExtractPlans(NProbDF)
    logger.debug("Finished extracting plans in needed format")

    NProbDF = planning_utils.find_sub_problems(config.domain_pddl_path, NProbDF, config.subproblems_dir)
    logger.debug("Finished creating sub problems")
    gen_utils.save_info_df_as_csv(NProbDF, config.csv_path)

    create_table.create_tables_add_df(NProbDF, config.domain_pddl_path, config)
    logger.debug("Finished creating tables")

    NProbDF.reset_index(drop=True, inplace=True)
    planning_utils.create_problem_images(NProbDF, config.img_dir, config.python_path, config.domain_pddl_path,
                                         config)
    logger.debug("Finished creating images")


    gen_utils.save_info_df_as_csv(NProbDF, config.csv_path)
    logger.debug("Finished data creation")



