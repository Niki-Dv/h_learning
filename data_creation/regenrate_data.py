"""
Script takes problem from some domain and uses:
1) domain.pddl
2) problem.pddl
3) plan for achieving the goal of problem (assumes plan actions are OK)
to create all the "mid-way" problems generated by doing each one of the actions in
plan after each plan action generates the problem achieved by that step

Each problem generated is saved to path (given as 4th parameter) as PDDL
and also saves to given excel file the path of new problem pddl and value
of its heuristic (which is given by the fact that we had plan and we take
one action each time).
"""

##############################################################################################
# Imports
import sys, os
import re
import h_config
config = h_config.config.get_config()

# add path to pddl parser project
if config.path_to_python_pddl not in sys.path:
    sys.path.append(config.path_to_python_pddl)

from pythonpddl import pddl


##############################################################################################
def delete_predicate(problem, predic_name, predic_args):
    """
    deletes from given problem the given predicate, changes problem itself.
    Args:
        problem: pddl problem variable
        predic_name: name of the predicate you want to delete
        predic_args: arguments of he predicate you want to delete in a list

    Returns:
        True if found the predicate and deleted, False otherwise
    """
    for pred in problem.initialstate:
        assert len(pred.subformulas) == 1, "I think it can only be 1, if not, notify me? "  # todo: check if correct
        if predic_name != pred.subformulas[0].name:  # todo: check why not exactly the same calibrate

            continue
        else:  # name matches, check args match
            pred_args = [arg.arg_name for arg in pred.subformulas[0].args.args]

            args_set = set(pred_args)
            if args_set == set(predic_args):
                problem.initialstate.remove(pred)
                return True

    # case couldn't find, something wrong
    return False


##############################################################################################
def create_new_problem_action(action_data, action, original_problem):
    """
    creates new problem where the problem is the given problem
    after making the action given (assumes that action provided is "OK")
    Args:
        action_data: extracts from the domain variable data about the action (parameters, pre-conds etc.)
        action: action that you want to make in given problem
        problem: pddl problem variable

    Returns:
        the problem achieved after the action on original problem is performed
    """
    problem = original_problem
    assert action_data.parameters.args.__len__() == (len(action) - 1), "Different num of params between" \
                                                                       " action passed and action data"
    # create dictionary between action data variable parameters and actual action parameters
    parm_dic = {}
    param_index = 1  # first place is for action name
    for argument in action_data.parameters.args:
        arg_name = argument.arg_name
        parm_dic[arg_name] = action[param_index]
        param_index += 1

    new_predics = []
    for effect in action_data.eff:
        action_op = effect.op
        for predic in effect.subformulas:
            predic_args = []
            for predic_argument in predic.args.args:
                var_name = predic_argument.arg_name
                predic_args.append(parm_dic[var_name])

            predic_data = [action_op, predic.name, predic_args]
            new_predics.append(predic_data)

    # change the problem init state pred
    for action_op, predic_name, predic_args in new_predics:
        if action_op == 'not':  # delete init state predicates
            # find the predicate to delete
            delete_predicate(problem, predic_name, predic_args)
        else:
            typed_arg_list = []
            for arg in predic_args:
                typed_arg_list.append(pddl.TypedArg(arg))
            formula_for_add = pddl.Formula([pddl.Predicate(predic_name, pddl.TypedArgList(typed_arg_list))])
            problem.initialstate.append(formula_for_add)

    return problem


##############################################################################################
def getActionsByArgs(plan_file):
    """
    reads line by line the plan file, splits the arguments
     and puts the action and its args into list (assumes that each line is action and action is not
     separated between two lines).
    Args:
        plan_file: file with the actions of the plan

    Returns:
        list of all the actions where each items is a list with action
        name and all the action arguments.
    """
    action_list = []
    with open(plan_file) as fd:
        line = fd.readline()
        while line:
            args = re.split(" |\(|\)", line)
            actions_by_args = []
            for arg in args:
                if arg != "" and arg != "\n":
                    actions_by_args.append(arg)

            action_list.append(actions_by_args)
            actions_by_args = []
            line = fd.readline()

    return action_list


##############################################################################################
def find_action_data(domain, action_name):
    """
    finds inside the domain variable the data of actions with action_name
    Args:
        domain: pddl.domain variable
        action_name: name of the action you want the data for

    Returns:
        pddl.Action variable with all the action data
    """
    for action_data in domain.actions:
        if action_data.name == action_name:
            return action_data


##############################################################################################
def flow(dom, prob, plan_actions, partial_new_problems_name):
    """
    iterates throught all the actions of plan one by one and
    generates the new problem, saves it to the path given(with partial name, adds count to
    the given name)
    Args:
        dom: pddl.Domain variable
        prob: pddl.Problem variable
        plan_actions: file with the actions of the plan
        partial_new_problems_name: partial path to save the new problems to.
                                    called partial because script add count of the action number
                                    that generated that problem.

    Returns:
        -
    """
    subproblem_paths = []
    for count, action in enumerate(plan_actions):
        action_data = find_action_data(dom, action[0])
        prob = create_new_problem_action(action_data, action, prob)
        new_prob_path = partial_new_problems_name + "_after_" + str(count+1) + "_steps.pddl"
        subproblem_paths.append(new_prob_path)
        fd = open(new_prob_path, "w")
        fd.write(prob.asPDDL())
        fd.close()
    return subproblem_paths

##############################################################################################
def gen_problem_sub_problems(domain_file_path, problem_file_path, plan_file_path, new_problems_path):
    (dom, prob) = pddl.parseDomainAndProblem(domain_file_path, problem_file_path)

    # create partial name path using path for problems and name of the given problem
    name_of_problem_file = problem_file_path.split(r'/')[-1]
    name_of_problem_file_wo_pddl = name_of_problem_file.split('.')[0]
    partial_new_problems_name = os.path.join(new_problems_path, name_of_problem_file_wo_pddl)

    # create actions list using the plan file
    plan_actions = getActionsByArgs(plan_file_path)
    if len(plan_actions) == 0:
        return plan_actions
    subproblem_paths = flow(dom, prob, plan_actions, partial_new_problems_name)
    return subproblem_paths


##############################################################################################
if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("-E- Usage: " + str(sys.argv[0]),
              "<domain pddl file path> <problem pddl file path> <plan for problem path> "
              "<path for for new problems>")
        sys.exit(1)

    domain_file_path = sys.argv[1]
    problem_file_path = sys.argv[2]
    plan_file_path = sys.argv[3]
    new_problems_path = sys.argv[4]

    gen_problem_sub_problems(domain_file_path, problem_file_path, plan_file_path, new_problems_path)


