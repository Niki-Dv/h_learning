# Imports
import sys, os
import numpy as np
import pandas as pd
import logging

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
package_path = os.path.join(curr_dir_path, '..')
if package_path not in sys.path:
    sys.path.append(package_path)
from data_creation import h_config

config = h_config.config.get_config()
# add path to pddl parser project
if config.path_to_python_pddl not in sys.path:
    sys.path.append(config.path_to_python_pddl)


logger = logging.getLogger()

from pythonpddl import pddl

domain_file_path = r"/media/sf_Project/Data_generator/Hiking/domain.pddl"

def extract(domain_file_path):
    dom= pddl.parseDomainOnly(domain_file_path)
    # create dictionary of domain predicates
    dom_predic_dict = {}
    for predic in dom.predicates:
        predic_name = predic.name.lower()
        args = []
        for typed_arg in predic.args.args:
            args.append(typed_arg.arg_name.lower())
            dom_predic_dict[predic_name] = 50
    type_dict = {}
    for type in dom.types.args:
        type_name = type.arg_name.lower()
        type_dict[type_name] = 10
    print(dom_predic_dict)
    print(type_dict)

if __name__ == '__main__':
    extract(domain_file_path)
    