# imports
import random
import subprocess
import pandas as pd
import numpy as np
import os, sys
import augment_problem_data

N=10
curr_dir = os.path.join(os.path.dirname(__file__))
data_path = os.path.join(curr_dir, 'data_for_domain')

# Preparing Rover domain needed inputs
generator_path = os.path.join(data_path, 'rovgen')
planner_path = r"/media/sf_Niki_Davarashvili_-_Project/fast_downward/fast-downward.py"
domain_path = os.path.join(data_path, 'domain.pddl')

ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
problems_dir = r"/media/sf_Niki_Davarashvili_-_Project/Data_generator/generated_problems"
subproblems_dir = r"/media/sf_Niki_Davarashvili_-_Project/Data_generator/generated_problems/generated_subproblems"
plans_dir = r"/media/sf_Niki_Davarashvili_-_Project/Data_generator/generated_problems"

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
def GenerateProblems(NProbDF, generator_path, planner_path):
    # create problem using the parameters in data frame
    for i in range(0, NProbDF.shape[0]):
        NProbDF.at[i, "id"] = i
        NProbDF.at[i, "from_id"] = i
        gen_val_list = NProbDF.iloc[i].to_list()[:-4]
        cmd_val_list = [np.array2string(val) for val in gen_val_list]
        cmd_val_list.insert(0, generator_path)
        cmd_line = " ".join(cmd_val_list)
        prob_out_path = os.path.join(problems_dir, "prob_" + i.__str__()+".pddl")
        fd = open(prob_out_path, "w")
        ProblemDescription = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
        fd.close()
        NProbDF.at[i, "problem.pddl"] = prob_out_path

    for i in range(0, NProbDF.shape[0]):
        # create plans for the given problem path in data frame :todo:Ive taken 'python' out of the arguments for the cmdline - check if needed
        cmd_line_args = [planner_path, domain_path, NProbDF.at[i, "problem.pddl"],
                         "--search \"astar(lmcut())\""]
        cmd_line = " ".join(cmd_line_args)

        plan_out_path = os.path.join(plans_dir, "plan_" + i.__str__())
        fd = open(plan_out_path, "w") #todo:the bug is that the plan is not printed to the file.. for some reason the plan file is empty
        PlanDescription = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
        fd.close()
        if PlanDescription.stderr:
            pass
        NProbDF.at[i, "plan"] = plan_out_path


###############################################################################################
# Function Description: Generates N problems of a certain domain using a given generator path and a list of its
#                      needed descriptor titles and their ranges. Stores all problems in a pandas dataframe
# Inputs: 1- N (integer) states the number of problems to generate 2 - Generator Path (string)
#        3- ProbDescriptorsInfo (list of lists) - each descriptor list includes the title of each descriptor (string)
#        and the min and max values possible for that descriptor
# Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
#           and the final column is the problem as generated,and each row represents a problem generated
###############################################################################################
def GenProblemsDataFrame(N, generator_path, planner_path, ProbDescriptorsInfo):
    DescriptorTitles = []
    #ProbDescriptorVals = []#todo: check if needed
    for i in range(len(ProbDescriptorsInfo)):
        DescriptorTitles.append(ProbDescriptorsInfo[i][0])
    DescriptorTitles.append("problem.pddl")
    DescriptorTitles.append("plan")
    DescriptorTitles.append("id")
    DescriptorTitles.append("from_id")
    list_generated_descriptor = []
    for i in range(N):
        ProbDescriptorVals = []
        for j in range(len(ProbDescriptorsInfo)):
            descr_value = random.randint(ProbDescriptorsInfo[j][1], ProbDescriptorsInfo[j][2])
            ProbDescriptorVals.append(descr_value)
            if j == (len(ProbDescriptorsInfo) - 1):
                ProbDescriptorVals += ["", "", i, "parent_problem"]
                list_generated_descriptor.append(ProbDescriptorVals)

    NProbDF = pd.DataFrame(list_generated_descriptor, columns=DescriptorTitles)
    GenerateProblems(NProbDF, generator_path, planner_path)
    return NProbDF

##############################################################################################
def ExtractPlan(NProbDF):
    import re
    plan_lengths = []
    good_lines = []
    for i in range(N):
        needed_data = []
        flag = False
        curr_plan_path = NProbDF.loc[NProbDF['id'] == i]["plan"].iloc[0]

        with open(curr_plan_path) as plan_f:
            line = plan_f.readline()

            while line:
                print(line)
                if flag==True:
                    good_lines.append(i)
                    fh = open(curr_plan_path, "w")
                    for plan_step in needed_data:
                        fh.write(plan_step)
                        fh.write("\n")
                    fh.close()
                    break

                if re.search("Actual search time", line) is not None:
                    line = plan_f.readline()
                    flag = True
                    print(i)
                    while line:
                        if re.search("Plan length:", line):
                            plan_length = re.findall("[0-9]+", line)
                            assert len(plan_length) == 1, "More than one number in plan length line"
                            plan_lengths.append(plan_length[0])
                            break
                        else:
                            line = line.split(' ')[:-1]
                            plan_line = "(" + " ".join(line) + ")"
                            needed_data.append(plan_line)

                        line = plan_f.readline()

                line = plan_f.readline()

    empty_lines = [i for i in range(N)]
    for good_line in good_lines:
        empty_lines.remove(good_line)

    NProbDF = NProbDF.drop([empty_lines])
    NProbDF["plan length"] = plan_lengths

    return NProbDF





##############################################################################################
def main(N, domain_file, generator_path, planner_path, ProbDescriptorsInfo, subproblems_dir):

    NProbDF = GenProblemsDataFrame(N, generator_path, planner_path, ProbDescriptorsInfo)
    NProbDF = ExtractPlan(NProbDF)
    #todo: check how the new_problems_path param is chosen - probably need to generate it inside the function
    NProbDF = augment_problem_data.main(domain_file, NProbDF, subproblems_dir)
    return NProbDF


##############################################################################################
if __name__ == '__main__':
    """
    if len(sys.argv) != 6:
        print("-E- Usage: " + str(sys.argv[0]),
              "<num of problems to generate> <domain pddl file path> <generator file path> <planner file path> "
              "<problem descriptors info> <path for for new problems>")
        sys.exit(1)

    N = sys.argv[1]
    domain_path = sys.argv[2]
    generator_path = sys.argv[3]
    planner_path = sys.argv[4]
    new_problems_path = sys.argv[5]
    """
    main(N, domain_path, generator_path, planner_path, ProbDescriptorsInfo, subproblems_dir)


