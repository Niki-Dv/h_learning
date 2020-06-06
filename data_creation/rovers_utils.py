import random
import pandas as pd
import os
import logging
import subprocess

logger = logging.getLogger()


###############################################################################################
def RoversGenProblemsParams(config):
    """
    Function Description: Generates N problems of a certain domain using a given generator path and a list of its
                     needed descriptor titles and their ranges

    Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
           and the final column is the problem as generated,and each row represents a problem generated
    """

    DescriptorTitles = []
    for i in range(len(config.ProbDescriptors)):
        DescriptorTitles.append(config.ProbDescriptors[i])

    DescriptorTitles.append("id")
    DescriptorTitles.append("from_id")

    generated_descriptors_list = []
    # order is: rovers, waypoints, objectives, camers, goals
    generator_limits = [1, 10, 10, 1, 10]
    for i in range(config.N):
        ProbDescriptorVals = [config.SEED+i]
        for j in range(len(generator_limits)):
            descr_value = random.randint(1, generator_limits[j])
            ProbDescriptorVals.append(descr_value)

        # number of waypoints has to be greater than number of rovers
        if ProbDescriptorVals[2] < ProbDescriptorVals[1]:
            ProbDescriptorVals[2] = ProbDescriptorVals[1]

        ProbDescriptorVals += [i, "parent_problem"]
        generated_descriptors_list.append(ProbDescriptorVals)

    NProbDF = pd.DataFrame(generated_descriptors_list, columns=DescriptorTitles)
    return NProbDF

###############################################################################################
def RoversGenerateProblems(NProbDF, config):
    """
    Create problems using external problem generator, add as rows
    """
    rows_to_delete = []
    NProbDF["problem"] = None
    processes = []
    # create problem using the parameters in data frame

    for i in range(0, NProbDF.shape[0]):
        NProbDF.at[i, "id"] = i
        NProbDF.at[i, "from_id"] = i
        cmd_val_list = NProbDF.iloc[i].to_list()[:6]
        cmd_val_list = [str(val) for val in cmd_val_list]
        cmd_val_list.insert(0, config.generator_path)
        cmd_line = " ".join(cmd_val_list)
        prob_out_path = os.path.join(config.problems_dir, "prob_" + i.__str__() + ".pddl")
        logger.debug("The cmd line is: {}".format(cmd_line))
        with open(prob_out_path, "w") as fd:
            proc = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
            proc.wait()

        if os.stat(prob_out_path).st_size == 0:
            rows_to_delete.append(i)
        else:
            NProbDF.at[i, "problem"] = prob_out_path

    NProbDF = NProbDF.drop(rows_to_delete)
    NProbDF.reset_index(drop=True, inplace=True)
    logger.debug('finished creating problems')

    return NProbDF