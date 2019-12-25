# imports
import random
import subprocess
import pandas as pd

###############################################################################################
#Function Description: Random generation of problem descriptors for the problem generator
#Inputs: 1- Minimum value (integer) of problem descriptor, 2- Maximum value (integer) of problem descriptor
#Output: 1- Descriptor (integer) for the relevant argument needed to invoke the problem generator
###############################################################################################
def GenRandDescriptor(MinDescriptor, MaxDescriptor):
    return random.randint(MinRov, MaxRov)

###############################################################################################
#Function Description: Activate external generator script using subprocess method, and adding
#                      the generated problem to the problem dataframe
#Inputs: 1- Generator Path (string) 2- ProblemDF (Pandas DataFrame) - includes the value of each descriptor
#Output: ProblemDF (Pandas DataFrame) - the input + the appended output of the generator
###############################################################################################
def GenSingleProblem(GeneratorPath, ProblemDF):
    ProblemDescription = subprocess.run([ProblemDescriptorVals, ProblemDF.at[0, "RovNum"], ProblemDF.at[0, "WayPointNum"],
                                         /ProblemDF.at[0, "ObjectivesNum"], ProblemDF.at[0, "CamerasNum"],
                                         /ProblemDF.at[0, "GoalsNum"]], stdout=subprocess.PIPE)
    ProblemDescriptionString = ProblemDescription.stdout.decode('utf-8')
    ProblemDF[0,"problem.pddl"]=ProblemDescriptionString

#Preparing Rover domain needed inputs
GeneratorPath= r"C:\Users\DELL\Technion\Niki Davarashvili - Project\Data generator\rovgen"
ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
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
ProbDescriptorsInfo= [["RovNum",MinRov,MaxRov],["WayPointNum",MinWayPoint,MaxWayPoint],["ObjectivesNum",MinObjectives,MaxObjectives],
                   /["CamerasNum",MinCameras,MaxCamera],["GoalsNum",MinGoals,MaxGoals]]
#df = pd.DataFrame(lst, index =['a', 'b', 'c', 'd', 'e', 'f', 'g'],
#                                              columns =['Names'])

###############################################################################################
#Function Description: Generates N problems of a certain domain using a given generator path and a list of its
#                      needed descriptor titles and their ranges. Stores all problems in a pandas dataframe
#Inputs: 1- N (integer) states the number of problems to generate 2 - Generator Path (string)
#        3- ProbDescriptorsInfo (list of lists) - each descriptor list includes the title of each descriptor (string)
#        and the min and max values possible for that descriptor
#Output: 1- NProbDF (Pandas DataFrame) - The columns of the dataframe are the problem descriptors and the
#           and the final column is the problem as generated,and each row represents a problem generated
###############################################################################################
def GenProblemsDataFrame(N,GeneratorPath, ProbDescriptorsInfo):
    DescriptorTitles=[]
    ProbDescriptorVals=[]
    for i in range(len(ProbDescriptorsInfo)):
        DescriptorTitles+=ProbDescriptorsInfo[i][0]
    DescriptorTitles+=["problem.pddl"]
    NProbDF = pd.DataFrame([])
    for i in range(N):
        for j in range (len(ProbDescriptorsInfo)):
            ProbDescriptorVals=[]
            ProbDescriptorVals += [GenRandDescriptor(ProbDescriptorsInfo[j][1], ProbDescriptorsInfo[j][2])]
            if (j==(len(ProbDescriptorsInfo)-1)):
                ProbDescriptorVals+=[""]
        SingleProblemDF = pd.DataFrame(ProbDescriptorVals, columns=DescriptorTitles)
        NProbDF.append(GenSingleProblem(GeneratorPath,SingleProblemDF)) #FIX ME
    return NProbDF

N=10
RoverDF = GenProblemsDataFrame(N,GeneratorPath, ProbDescriptorsInfo)

