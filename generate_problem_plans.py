# imports
import random
import subprocess
import pandas as pd
import numpy as np
import os, sys
import augment_problem_data
import time
import codecs

N = 10
curr_dir = os.path.join(os.path.dirname(__file__))
data_path = os.path.join(curr_dir, 'data_for_domain')

# Preparing Rover domain needed inputs
generator_path = os.path.join(data_path, 'rovgen')
planner_path = r"/media/sf_Project/fast_downward/fast-downward.py"
domain_path = os.path.join(data_path, 'domain.pddl')

ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
problems_dir = r"/media/sf_Project/Data_generator/generated_problems"
subproblems_dir = r"/media/sf_Project/Data_generator/generated_problems/generated_subproblems"
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
ProbDescriptorsInfo = [["Seed", MinSeed, MaxSeed], ["RovNum", MinRov, MaxRov],
                       ["WayPointNum", MinWayPoint, MaxWayPoint],
                       ["ObjectivesNum", MinObjectives, MaxObjectives], \
                       ["CamerasNum", MinCameras, MaxCameras], ["GoalsNum", MinGoals, MaxGoals]]


###############################################################################################
def GenerateProblems(NProbDF, generator_path, planner_path):
    """
    Activate external problems generator code and add
    the generated problem to the problem data frame
    :param NProbDF:
    :param generator_path:
    :param planner_path:
    :return:
    """

    processes = []
    # create problem using the parameters in data frame
    for i in range(0, NProbDF.shape[0]):
        NProbDF.at[i, "id"] = i
        NProbDF.at[i, "from_id"] = i
        gen_val_list = NProbDF.iloc[i].to_list()[:-4]
        cmd_val_list = [np.array2string(val) for val in gen_val_list]
        cmd_val_list.insert(0, generator_path)
        cmd_line = " ".join(cmd_val_list)
        prob_out_path = os.path.join(problems_dir, "prob_" + i.__str__() + ".pddl")
        with open(prob_out_path, "w") as fd:
            ProblemDescription = subprocess.Popen(cmd_line, stdout=fd, stderr=subprocess.PIPE, shell=True)
            processes.append(ProblemDescription)
        NProbDF.at[i, "problem"] = prob_out_path

    for process in processes:
        process.wait()

    processes = []
    # for each problem, create plan by planner
    for i in range(0, NProbDF.shape[0]):
        plan_out_path = os.path.join(plans_dir, "plan_" + i.__str__() + ".txt")
        cmd_line_args = ['python', planner_path,"--plan-file", plan_out_path, domain_path, NProbDF.at[i, "problem"],
                         "--search \"astar(lmcut())\""]
        cmd_line = " ".join(cmd_line_args)
        sub_res = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        processes.append(sub_res)
        NProbDF.at[i, "plan"] = plan_out_path

    for process in processes:
        process.wait()


###############################################################################################
def GenProblemsDataFrame(N, generator_path, planner_path, ProbDescriptorsInfo):
    """
    Function Description: Generates N problems of a certain domain using a given generator path and a list of its
                     needed descriptor titles and their ranges
    Inputs: 1- N (integer) states the number of problems to generate 2 - Generator Path (string)
        3- ProbDescriptorsInfo (list of lists) - each descriptor list includes the title of each descriptor (string)
        and the min and max values possible for that descriptor
    Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
           and the final column is the problem as generated,and each row represents a problem generated
    """

    DescriptorTitles = []
    for i in range(len(ProbDescriptorsInfo)):
        DescriptorTitles.append(ProbDescriptorsInfo[i][0])
    DescriptorTitles.append("problem")
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
    """
    Extracts the plan itself from all the plan data that is created by planner
    and overrides the original plan generated by planner
    """
    import re
    time.sleep(1)
    plan_lengths = []
    correct_plans = []
    for index, row in NProbDF.iterrows():
        needed_data = []
        curr_plan_path = row["plan"]
        plan_exist = False
        print("STARTED WITH PLAN: " + str(index))
        if not os.path.exists(curr_plan_path):
            print("no plan for: " + str(index))
            continue
        with open(curr_plan_path, 'r') as plan_f:
            for cnt, line in enumerate(plan_f):
                if re.search("cost = ", line, re.IGNORECASE):
                    plan_length = re.findall("[0-9]+", line)
                    plan_lengths.append(plan_length[0])
                    plan_exist = True
                    print("FOUND PLAN FOR: " + str(index))
                    break

                needed_data.append(line)
                print("NIKI")
                print(line)
                print("NIKI OUT")

        if plan_exist:
            correct_plans.append(index)
            with open(curr_plan_path, "w") as fh:
                for plan_step in needed_data:
                    fh.write(plan_step)
                print("WROTE PLAN FOR: " + str(index))
                print("in path: " + curr_plan_path)

    corrupt_plans = [i for i in range(N)]
    for correct_plan in correct_plans:
        corrupt_plans.remove(correct_plan)

    NProbDF = NProbDF.drop(corrupt_plans)
    NProbDF["plan length"] = plan_lengths

    return NProbDF


##############################################################################################
def main(N, domain_file, generator_path, planner_path, ProbDescriptorsInfo, subproblems_dir):
    NProbDF = GenProblemsDataFrame(N, generator_path, planner_path, ProbDescriptorsInfo)
    NProbDF = ExtractPlan(NProbDF)
    # todo: check how the new_problems_path param is chosen - probably need to generate it inside the function
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
