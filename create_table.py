# Imports
import sys, os
import numpy as np
import pandas as pd
import logging
import h_config
config = h_config.config.get_config()
logger = logging.getLogger()


# add path to pddl parser project
if config.path_to_python_pddl not in sys.path:
    sys.path.append(config.path_to_python_pddl)

from pythonpddl import pddl

##############################################################################################
def problem_to_dict(domain_file_path, problem_path):
    (dom, prob) = pddl.parseDomainAndProblem(domain_file_path, problem_path)
    # create dictionary of the objects
    obj_dict = {}
    for object in prob.objects.args:
        obj_type = object.arg_type.lower()
        obj_name = object.arg_name.lower()
        if obj_type in obj_dict.keys():
            obj_dict[obj_type].append(obj_name)
        else:
            obj_dict[obj_type] = [obj_name]

    # create dictionary of the init predicates
    init_predic_dict = {}
    for predic in prob.initialstate:
        predic_name = predic.subformulas[0].name.lower()
        args = []
        for typed_arg in predic.subformulas[0].args.args:
            args.append(typed_arg.arg_name.lower())

        if predic_name in init_predic_dict.keys():
            init_predic_dict[predic_name].append(args)
        else:
            init_predic_dict[predic_name] = [args]

    # create goal
    goal_dict = {}
    for goal in prob.goal.subformulas:
        goal_pred_name = goal.subformulas[0].name.lower()
        args = []
        for typed_arg in goal.subformulas[0].args.args:
            args.append(typed_arg.arg_name.lower())

        if goal_pred_name in goal_dict.keys():
            goal_dict[goal_pred_name].append(args)
        else:
            goal_dict[goal_pred_name] = [args]

    # create dictionary of domain predicates
    dom_predic_dict = {}
    for predic in dom.predicates:
        predic_name = predic.name.lower()
        args = []
        for typed_arg in predic.args.args:
            args.append(typed_arg.arg_name.lower())

        if predic_name in dom_predic_dict.keys():
            dom_predic_dict[predic_name].append(args)
        else:
            dom_predic_dict[predic_name] = [args]

    # get objects list
    dom_object_types = [arg.arg_name.lower() for arg in dom.types.args]

    return obj_dict, init_predic_dict, goal_dict, dom_predic_dict, dom_object_types

##############################################################################################
def create_object_name_to_columns_dict(obj_dict, dom_object_types):
    #look like this: dict = {"rover0": 0, "rover1": 1, 'waypoint1': 6}
    obj_limit_dict = h_config.Objects_limit_dict.copy()
    columns_idxs = 0
    obj_col_dict = {}
    for object_type in dom_object_types:
        if object_type in obj_dict.keys():
            for prob_obj in obj_dict[object_type]:
                # check if limit reached
                obj_limit_dict[object_type] -= 1
                if obj_limit_dict[object_type] < 0:
                    logger.error("too many object of type: {}".format(object_type))
                    sys.exit(-1)
                obj_col_dict[prob_obj] = columns_idxs
                columns_idxs += 1

        while obj_limit_dict[object_type] != 0:
            obj_limit_dict[object_type] -= 1
            columns_idxs += 1

    obj_col_dict[h_config.GOAL_COL_NAME] = columns_idxs
    return obj_col_dict



##############################################################################################
def create_table(init_predic_dict, goal_dict, dom_predic_dict, obj_col_dict):
    rows_idxs = 0
    predic_lim_dict = h_config.predic_limit_dict.copy()
    table = np.zeros((h_config.num_predic, h_config.num_objects + 1))
    for predic_name in dom_predic_dict.keys():
        if predic_name in init_predic_dict.keys():
            for predic_args in init_predic_dict[predic_name]:
                predic_lim_dict[predic_name] -= 1
                if predic_lim_dict[predic_name] < 0:
                    logger.error("too many predicates of type: {}".format(predic_name))
                    sys.exit(-1)

                for arg in predic_args:
                    arg_col_idxs = obj_col_dict[arg]
                    table[rows_idxs, arg_col_idxs] = 1
                # increase row idx
                rows_idxs += 1

        if predic_name in goal_dict.keys():
            for predic_args in goal_dict[predic_name]:
                predic_lim_dict[predic_name] -= 1
                if predic_lim_dict[predic_name] < 0:
                    logger.error("too many predicates of type: {}".format(predic_name))
                    sys.exit(-1)

                for arg in predic_args:
                    arg_col_idxs = obj_col_dict[arg]
                    table[rows_idxs, arg_col_idxs] = 1
                    goal_col_idx = obj_col_dict[h_config.GOAL_COL_NAME]
                    table[rows_idxs, goal_col_idx] = 1
                # increase row idx
                rows_idxs += 1
                
        while predic_lim_dict[predic_name] != 0:
            predic_lim_dict[predic_name] -= 1
            rows_idxs += 1

    return table

##############################################################################################
def main(domain_file_path, csv_path, table_path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        obj_dict, init_predic_dict, goal_dict, dom_predic_dict, dom_object_types = problem_to_dict(domain_file_path, row["problem"])
        obj_col_dict = create_object_name_to_columns_dict(obj_dict, dom_object_types)
        table = create_table(init_predic_dict, goal_dict, dom_predic_dict, obj_col_dict)


##############################################################################################
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("-E- Usage: " + str(sys.argv[0]),
              "<domain pddl file path> <path to csv with 'problem' column> <path for result table> ")
        sys.exit(1)

    domain_file_path = sys.argv[1]
    csv_path = sys.argv[2]
    table_path = sys.argv[3]

    main(domain_file_path, csv_path, table_path)

# cmd line needed: /media/sf_Project/Data_generator/domain.pddl /media/sf_Project/Data_generator/generated_problems/csv_dir/total.csv /media/sf_Project/Data_generator/generated_problems/tables

