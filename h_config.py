import os, sys
import logging

###############################################################################################
# configs for the generated problems


N = 300
###############################################################################################

class config:
    __conf = None

    def __init__(self):
        # set python3 and above path (or alias for your computer)
        self.python_path = 'python3.6'

        self.N = N

        curr_dir = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(curr_dir, 'data_gen')

        # define paths
        self.generator_path = os.path.join(data_path, 'rovgen')
        self.domain_pddl_path = os.path.join(data_path, 'domain.pddl')

        project_dir_path = r"/media/sf_Project/"
        self.logger_path = os.path.join(project_dir_path, r"Data_generator/log")

        self.planner_path = os.path.join(project_dir_path, r"fast_downward/fast-downward.py")

        self.problems_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems/original_problems/problems")
        self.subproblems_dir = os.path.join(project_dir_path,
                                            r"Data_generator/generated_problems/sub_problems/generated_subproblems")
        self.plans_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems/plans/plans")
        self.img_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems/images/images")
        self.csv_path = os.path.join(project_dir_path, r"Data_generator/generated_problems/csv_dir/info")

        # path to project of pddl parser, more information: https://github.com/karpase/pythonpddl
        self.path_to_python_pddl = os.path.join(project_dir_path, "shared_with_ubunto/pythonpddl")

        # path to delfi planner code image creater
        self.image_creater_path = '"' + os.path.join(project_dir_path,
                                               r"delfi code/ipc2018-classical-team23-ea3feb7f51a3/image-only.py") + '"'


        # names of problem params
        self.ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
        self.ProbDescriptorsInfo = ProbDescriptorsInfo

        self.planner_search_flag = "--search \"astar(add())\""

        self.plan_finding_timeout = 10
        config.__conf = self

    @staticmethod
    def get_config():
        if config.__conf is None:
            config.__conf = config()

        return config.__conf

###################################################################################################################
def define_logger(logger, logger_path):
    """
    Defines logger path, format etc.
    :param logger: object of logger return from logging.getLogger()
    :param logger_path: path for logger file
    :return: -
    """
    if len(logger.handlers) == 0:
        # format of logging messages
        formatter = logging.Formatter('%(asctime)s | %(levelname).1s'
                                      ' | %(filename)s#%(funcName)s:%(lineno)04d | %(message)s')
        # here you set logger level
        logger.setLevel(logging.DEBUG)

        # set file handler of logger
        fh = logging.FileHandler(logger_path)
        # set format for file handler
        fh.setFormatter(formatter)
        # set level for file logging
        fh.setLevel(logging.DEBUG)

        # config the logger so that all messages above error level also printed to stderr
        ch = logging.StreamHandler(sys.stderr)
        # set format for stderr stream handler
        ch.setLevel(logging.ERROR)
        # set format for stderr stream handler
        ch.setFormatter(formatter)

        # add both handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)


# heuristic setup of problem difficulties based on the 40 problems that are shown at planning.domains editor under the
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