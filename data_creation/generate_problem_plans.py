# imports
from datetime import datetime
import os, sys
import logging

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.join(curr_dir_path, '..')
if package_path in sys.path:
    sys.path.append(package_path)

from data_creation import gen_utils, planning_utils, h_config
logger = logging.getLogger()

config = h_config.config().get_config()

##############################################################################################
def main():
    h_config.define_logger(logger, config.logger_path)
    logger.info("Started generating problems and plans script")
    gen_utils.add_date_to_paths(config)
    gen_utils.create_dirs([config.plans_dir, config.problems_dir, config.subproblems_dir, config.img_dir])
    logger.debug("Finished preparing directories and paths, starting generating data")

    NProbDF = planning_utils.GenProblemsParams(config)
    NProbDF = planning_utils.GenerateProblems(NProbDF)
    logger.debug("Finished creating problems")
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

    planning_utils.create_problem_images(NProbDF, config.img_dir, config.python_path, config.domain_pddl_path,
                          config)
    logger.debug("Finished creating images")
    gen_utils.save_info_df_as_csv(NProbDF, config.csv_path)
    logger.debug("Finished data creation")
    
    return NProbDF


if __name__ == '__main__':
    main()
