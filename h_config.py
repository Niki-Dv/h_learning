import os

###############################################################################################
# configs for the generated problems

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


###############################################################################################

class config:
    __conf = None

    def __init__(self):
        # set python3 and above path (or alias for your computer)
        self.python_path = 'python'

        self.N = 10

        curr_dir = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(curr_dir, 'data_gen')

        # define paths
        self.generator_path = os.path.join(data_path, 'rovgen')
        self.domain_pddl_path = os.path.join(data_path, 'domain.pddl')

        project_dir_path = r"/media/sf_Project/"
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
        self.ProbDescriptorsInfo = [["Seed", MinSeed, MaxSeed], ["RovNum", MinRov, MaxRov],
                                    ["WayPointNum", MinWayPoint, MaxWayPoint],
                                    ["ObjectivesNum", MinObjectives, MaxObjectives], \
                                    ["CamerasNum", MinCameras, MaxCameras], ["GoalsNum", MinGoals, MaxGoals]]

        self.planner_search_flag = "--search \"astar(add())\""

        config.__conf = self

    @staticmethod
    def get_config():
        if config.__conf is None:
            config.__conf = config()

        return config.__conf
