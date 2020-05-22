import os, sys
import logging


class config:
    __conf = None

    def __init__(self):
        self.data_gen_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator"
        self.results_dir = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results"
        self.saved_models_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results\saved_models"
        self.data_csv_path = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\generated_problems\csv_dir\info_28_03_2020_17_45_22.csv"

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
        logger.setLevel(logging.INFO)

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