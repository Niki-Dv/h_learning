import random
import pandas as pd



###############################################################################################
def GenProblemsParams(config):
    """
    Function Description: Generates N problems of a certain domain using a given generator path and a list of its
                     needed descriptor titles and their ranges

    Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
           and the final column is the problem as generated,and each row represents a problem generated
    """

    DescriptorTitles = []
    for i in range(len(config.ProbDescriptorsInfo[0])):
        DescriptorTitles.append(config.ProbDescriptorsInfo[0][i][0])
    DescriptorTitles.append("problem")
    DescriptorTitles.append("plan")
    DescriptorTitles.append("id")
    DescriptorTitles.append("from_id")
    list_generated_descriptor = []

    for i in range(config.N):
        curr_diff_lvl = 0 # todo: change back
        ProbDescriptorVals = []
        for j in range(len(config.ProbDescriptorsInfo[0])):
            descr_value = random.randint(config.ProbDescriptorsInfo[curr_diff_lvl][j][1],
                                         config.ProbDescriptorsInfo[curr_diff_lvl][j][2])
            ProbDescriptorVals.append(descr_value)
            if j == (len(config.ProbDescriptorsInfo[0]) - 1):
                ProbDescriptorVals += ["", "", i, "parent_problem"]
                list_generated_descriptor.append(ProbDescriptorVals)

    NProbDF = pd.DataFrame(list_generated_descriptor, columns=DescriptorTitles)
    return NProbDF

###############################################################################################
#heuristic setup of problem difficulties based on the 40 problems that are shown at planning.domains editor under the
# 2006 rovers domain generator, which matches the domain.pddl that we are using
# Data will consist of 100 original problems, 10 for each difficulty level. Each difficulty level is created by randint
# descriptor generation which is based on AvgDescriptorVals[#difficulty_level_number] list. and each problem is
# generated based on MinMaxDescriptorPairs lists that indicate the range to sample the descriptor from in each
# difficulty level
# The rest of the data will be taken from the augmentation process that is done by regenerate_data.py script
NUM_OF_DESCRIPTORS = 6  # its actually 5 descriptors and seed is always the same for the generator cmd_line
NUM_OF_DIFFICULTY_LEVELS = 10

AvgDescriptorVals = [[] for i in range(NUM_OF_DIFFICULTY_LEVELS)]
AvgDescriptorVals[0] = [2, 3, 4, 3, 3, 3]
AvgDescriptorVals[1] = [2, 4, 4, 4, 4, 5]
AvgDescriptorVals[2] = [2, 5, 4, 5, 5, 7]
AvgDescriptorVals[3] = [2, 6, 4, 6, 6, 10]
AvgDescriptorVals[4] = [2, 7, 20, 7, 7, 14]
AvgDescriptorVals[5] = [2, 8, 25, 8, 10, 17]
AvgDescriptorVals[6] = [2, 9, 35, 9, 13, 20]
AvgDescriptorVals[7] = [2, 10, 50, 10, 15, 25]
AvgDescriptorVals[8] = [2, 12, 70, 12, 17, 40]
AvgDescriptorVals[9] = [2, 14, 90, 14, 20, 60]
DescriptorDelta = [[] for i in range(NUM_OF_DIFFICULTY_LEVELS)]

for j, obj in enumerate(DescriptorDelta):
    if j == 0:
        DescriptorDelta[j] = [0, 2, 2, 2, 2, 2]
    else:
        for k in range(NUM_OF_DESCRIPTORS):
            if k == 0:
                obj.append(0)  # seed number is always 2
            else:
                obj.append(AvgDescriptorVals[j][k]-AvgDescriptorVals[j-1][k])
# for each difficulty level there is a min & max descriptor value
MinDescriptors = [[] for i in range(NUM_OF_DIFFICULTY_LEVELS)]
MaxDescriptors = [[] for i in range(NUM_OF_DIFFICULTY_LEVELS)]
for i, MinDesc in enumerate(MinDescriptors):
    for j in range(NUM_OF_DESCRIPTORS):
        MinDesc.append(AvgDescriptorVals[i][j]-DescriptorDelta[i][j])
for i, MaxDesc in enumerate(MaxDescriptors):
    for j in range(NUM_OF_DESCRIPTORS):
        MaxDesc.append(AvgDescriptorVals[i][j]+DescriptorDelta[i][j])

ProbDescriptorsInfo = [[[] for j in range(NUM_OF_DESCRIPTORS)] for i in range(NUM_OF_DIFFICULTY_LEVELS)]
DescriptorTitles = ["Seed", "RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
for i, DiffLevel in enumerate(ProbDescriptorsInfo):
    for j, probDesc in enumerate(DiffLevel):
        probDesc.append(DescriptorTitles[j])
        probDesc.append(MinDescriptors[i][j])
        probDesc.append(MaxDescriptors[i][j])