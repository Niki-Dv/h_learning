"""
Script receives:
1) domain.pddl
2) NProbDF - a pandas dataframe that holds paths to N problems and their plans generated from the given domain
3) new_problems_path
The script iterates through the NProbDF dataframe created by the "generate_problem_plans.py" and sends the paths of each
problem and its corresponding plan to the "regenerate_data.py" script that augments data by create all the "mid-way"
problems generated by doing each one of the actions in plan after each plan action generates the problem achieved by
that step. for achieving the goal of problem (assumes plan actions are OK).

Each subproblem created by regenerate data is then added to NProbDF which is then returned to the
"generate_problem_plans.py" script
"""
import regenrate_data
import pandas as pd
import sys


##############################################################################################
def add_subproblems_to_df(df, subproblem_paths, df_parent_row):
    """
    adds given sub problem path to problems data frame
    :param subproblem_paths: paths to all the sub problems
    :param df_parent_row:
    :return:
    """
    prob_row = df_parent_row
    df_insert = pd.DataFrame([prob_row]*len(subproblem_paths), columns=df.columns).reset_index(drop=True)

    # stating that all rows that will be added to the df are of subproblems
    df_insert["id"] = "sub"

    sub_prob_index = 1
    for index, row in df_insert.iterrows():
        # assigning the subproblem path to each line
        df_insert.at[index, "problem"] = subproblem_paths[index]
        df_insert.at[index, "plan length"] = int(row["plan length"]) - sub_prob_index
        sub_prob_index += 1

    df = pd.concat([df, df_insert]).reset_index(drop=True)

    return df


##############################################################################################
def main(domain_file, NProbDF, new_problems_path):
    for index, row in NProbDF.iterrows():
        curr_prob_path = row["problem"]
        curr_plan_path = row["plan"]
        curr_prob_id = row["id"]
        subproblem_paths = regenrate_data.main(domain_file, curr_prob_path, curr_plan_path, new_problems_path)
        if len(subproblem_paths) == 0:
            continue
        NProbDF = add_subproblems_to_df(NProbDF, subproblem_paths, row)

    return NProbDF


##############################################################################################
if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("-E- Usage: " + str(sys.argv[0]),
              "<domain pddl file path> <problem pddl file path> <plan for problem path> "
              "<path for for new problems>")
        sys.exit(1)
    N = sys.argv[1]
    domain_file = sys.argv[2]
    NProbDF = sys.argv[3]
    new_problems_path = sys.argv[4]

    main(N, domain_file, NProbDF, new_problems_path)
