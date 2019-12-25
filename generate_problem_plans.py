# imports
import random
import subprocess
import pandas as pd
import numpy as np
import os, sys

curr_dir = os.path.join(os.path.dirname(__file__))
data_path = os.path.join(curr_dir, 'data_for_domain')

# Preparing Rover domain needed inputs
GeneratorPath = os.path.join(data_path, 'rovgen')
PlannerPath = r"/media/sf_Project/fast_downward/fast-downward.py"
domain_path = os.path.join(data_path, 'domain.pddl')

ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
problems_dir = r"/media/sf_Project/Data_generator/generated_problems"
plans_dir = r"/media/sf_Project/Data_generator/generated_problems"

MinSeed = 2
MaxSeed = 2
MinRov = 1
MaxRov = 4
MinWayPoint = 1
MaxWayPoint = 4
MinObjectives = 1
MaxObjectives = 4
MinCameras = 1
MaxCameras = 4
MinGoals = 1
MaxGoals = 4
ProbDescriptorsInfo = [["Seed", MinSeed, MaxSeed], ["RovNum", MinRov, MaxRov], ["WayPointNum", MinWayPoint, MaxWayPoint],
                       ["ObjectivesNum", MinObjectives, MaxObjectives],\
                       ["CamerasNum", MinCameras, MaxCameras], ["GoalsNum", MinGoals, MaxGoals]]



###############################################################################################
# Function Description: Activate external generator script using subprocess method, and adding
#                      the generated problem to the problem dataframe
# Inputs: 1- Generator Path (string) 2- ProblemDF (Pandas DataFrame) - includes the value of each descriptor
# Output: ProblemDF (Pandas DataFrame) - the input + the appended output of the generator
###############################################################################################
def GenerateProblems(NProbDF, GeneratorPath):
    # create problem using the parameters in data frame
    for i in range(0, NProbDF.shape[0]):
        gen_val_list = NProbDF.iloc[i].to_list()[:-2]

        cmd_val_list = [np.array2string(val) for val in gen_val_list]
        cmd_val_list.insert(0, GeneratorPath)
        cmd_line = " ".join(cmd_val_list)
        prob_out_path = os.path.join(problems_dir, "prob_" + i.__str__()+".pddl")
        fd = open(prob_out_path, "w")
        ProblemDescription = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
        fd.close()
        NProbDF.at[i, "problem.pddl"] = prob_out_path

    for i in range(0, NProbDF.shape[0]):
        # create problem using the parameters in data frame
        cmd_line_args = ['python', PlannerPath, domain_path, NProbDF.at[i, "problem.pddl"],
                         "--search \"astar(lmcut())\""]
        cmd_line = " ".join(cmd_line_args)

        plan_out_path = os.path.join(plans_dir, "plan_" + i.__str__())
        fd = open(plan_out_path, "w")
        PlanDescription = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
        fd.close()
        if PlanDescription.stderr:
            pass
        NProbDF.at[i, "plan"] = plan_out_path
    a=3

###############################################################################################
# Function Description: Generates N problems of a certain domain using a given generator path and a list of its
#                      needed descriptor titles and their ranges. Stores all problems in a pandas dataframe
# Inputs: 1- N (integer) states the number of problems to generate 2 - Generator Path (string)
#        3- ProbDescriptorsInfo (list of lists) - each descriptor list includes the title of each descriptor (string)
#        and the min and max values possible for that descriptor
# Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
#           and the final column is the problem as generated,and each row represents a problem generated
###############################################################################################
def GenProblemsDataFrame(N, GeneratorPath, ProbDescriptorsInfo):
    DescriptorTitles = []
    ProbDescriptorVals = []
    for i in range(len(ProbDescriptorsInfo)):
        DescriptorTitles.append(ProbDescriptorsInfo[i][0])
    DescriptorTitles.append("problem.pddl")
    DescriptorTitles.append("plan")
    list_generated_descriptor = []
    for i in range(N):
        ProbDescriptorVals = []
        for j in range(len(ProbDescriptorsInfo)):
            descr_value = random.randint(ProbDescriptorsInfo[j][1], ProbDescriptorsInfo[j][2])
            ProbDescriptorVals.append(descr_value)
            if j == (len(ProbDescriptorsInfo) - 1):
                ProbDescriptorVals += ["", ""]
                list_generated_descriptor.append(ProbDescriptorVals)

    NProbDF = pd.DataFrame(list_generated_descriptor, columns=DescriptorTitles)
    GenerateProblems(NProbDF, GeneratorPath)
    return NProbDF


if __name__ == '__main__':
    N = 10
    RoverDF = GenProblemsDataFrame(N, GeneratorPath, ProbDescriptorsInfo)
