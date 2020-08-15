import os, sys
import logging
import random

###############################################################################################
class config:
    __conf = None

    def __init__(self):
        # set python3 and above path (or alias for your computer)
        self.python_path = 'python3.6'

        # number of original problems to generate
        self.N = 2000
        
        self.project_dir_path = r"/media/sf_Project/"
        data_creation_path = os.path.join(self.project_dir_path, r"Data_generator/generated_problems/test")

        self.logger_path = os.path.join(self.project_dir_path, r"Data_generator/log")

        self.planner_path = os.path.join(self.project_dir_path, r"fast_downward/fast-downward.py")

        self.problems_dir = os.path.join(self.project_dir_path, data_creation_path, r"original_problems/problems")
        self.subproblems_dir = os.path.join(self.project_dir_path, data_creation_path,
                                            r"sub_problems/generated_subproblems")
        self.plans_dir = os.path.join(self.project_dir_path, data_creation_path, r"plans/plans")
        self.img_dir = os.path.join(self.project_dir_path, data_creation_path, r"images/images")
        self.tables_dir = os.path.join(self.project_dir_path, data_creation_path, r"tables/tables")
        self.csv_path = os.path.join(self.project_dir_path, data_creation_path, r"csv_dir/info")

        # path to project of pddl parser, more information: https://github.com/karpase/pythonpddl
        self.path_to_python_pddl = os.path.join(self.project_dir_path, "shared_with_ubunto/pythonpddl")

        # path to delfi planner code image creater
        self.image_creater_path = '"' + os.path.join(self.project_dir_path,
                                               r"delfi code/ipc2018-classical-team23-ea3feb7f51a3/image-only.py") + '"'


        self.planner_search_flag = "--search \"astar(ff)\""

        self.plan_finding_timeout = 15
        self.SEED = random.randint(0, round(sys.maxsize/2))
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

###########################################################################################################
class RoversConfig(config):
    Name = "rovers"
    def __init__(self):
        super().__init__()
        # define paths
        curr_dir = os.path.join(os.path.dirname(__file__))
        domain_data_path = os.path.join(curr_dir, "..", 'data_gen')
        self.generator_path = os.path.join(domain_data_path, 'rovgen')
        self.domain_pddl_path = os.path.join(domain_data_path, 'domain.pddl')

        # names of problem params
        self.ProbDescriptors = ["Seed", "RovNum", "WayPointNum", "ObjectivesNum", "CamerasNum", "GoalsNum"]
        # Table configs
        self.Objects_limit_dict = {'rover': 6, 'waypoint': 15, 'store': 6, 'camera': 6, 'mode': 6, 'lander': 6,
                              'objective': 15}
        self.predic_limit_dict = {'at': 20, 'at_lander': 15, 'can_traverse': 50, 'equipped_for_soil_analysis': 15,
                             'equipped_for_rock_analysis': 15, 'equipped_for_imaging': 15, 'empty': 10,
                             'have_rock_analysis': 10,
                             'have_soil_analysis': 10, 'full': 10, 'calibrated': 10, 'supports': 10,
                             'available': 10,
                             'visible': 50, 'have_image': 15, 'communicated_soil_data': 15,
                             'communicated_rock_data': 15,
                             'communicated_image_data': 15, 'at_soil_sample': 15, 'at_rock_sample': 15,
                             'visible_from': 50,
                             'store_of': 20, 'calibration_target': 15, 'on_board': 15, 'channel_free': 15}



        self.num_objects = 0
        for object_limit in self.Objects_limit_dict.keys():
            self.num_objects += self.Objects_limit_dict[object_limit]

        self.num_predic = 0
        for predic_limit in self.predic_limit_dict.keys():
            self.num_predic += self.predic_limit_dict[predic_limit]

        self.GOAL_COL_NAME = 'goal'

###########################################################################################################
class HikingConfig(config):
    Name = "hiking"
    def __init__(self):
        super().__init__()
        # define paths
        self.generator_path = os.path.join(self.project_dir_path,"Data_generator", 'Hiking', "generator.py")
        self.domain_pddl_path = os.path.join(self.project_dir_path,"Data_generator", 'Hiking', "domain.pddl")

        # names of problem params
        self.ProbDescriptors = ["couples", "cars", "places"]
        # Table configs
        self.Objects_limit_dict = {'car': 10, 'tent': 10, 'person': 20, 'couple': 10, 'place': 10}
        self.predic_limit_dict = {'at_tent': 40, 'at_person': 40, 'at_car': 40, 'partners': 40, 'up': 40,
                                  'down': 40, 'walked': 40, 'next': 40}
        self.num_objects = 0
        for object_limit in self.Objects_limit_dict.keys():
            self.num_objects += self.Objects_limit_dict[object_limit]

        self.num_predic = 0
        for predic_limit in self.predic_limit_dict.keys():
            self.num_predic += self.predic_limit_dict[predic_limit]

        self.GOAL_COL_NAME = 'goal'

###########################################################################################################












