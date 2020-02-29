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
    def __init__(self):
        self.N = 10

        curr_dir = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(curr_dir, 'data_gen')

        # define paths
        self.generator_path = os.path.join(data_path, 'rovgen')
        self.domain_pddl_path = os.path.join(data_path, 'domain.pddl')

        project_dir_path = r"/media/sf_Project/"
        self.planner_path = os.path.join(project_dir_path, r"fast_downward/fast-downward.py")
        self.problems_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems")
        self.subproblems_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems/generated_subproblems")
        self.plans_dir = os.path.join(project_dir_path, r"Data_generator/generated_problems/plans")

        # names of problem params
        self.ProbDescriptors = ["RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
        self.ProbDescriptorsInfo = [["Seed", MinSeed, MaxSeed], ["RovNum", MinRov, MaxRov],
                               ["WayPointNum", MinWayPoint, MaxWayPoint],
                               ["ObjectivesNum", MinObjectives, MaxObjectives], \
                               ["CamerasNum", MinCameras, MaxCameras], ["GoalsNum", MinGoals, MaxGoals]]
        self.conf = None

    def get_config(self):
        if self.conf is None:
            self.conf = config()

        return self.conf
