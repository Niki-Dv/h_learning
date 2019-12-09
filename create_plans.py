import sys, os
import urllib2, json, sys
import re

path_to_api = r"C:\dev\project\classical-domains\api-tools"
sys.path.append(path_to_api)
import planning_domains_api as api

domain_name = "rovers"
domain_id = api.find_domains(domain_name)[0]["domain_id"]  # get second rovers problem id (rover-02)
problems = api.get_problems(domain_id)

path_for_plans = r"C:\dev\h_learning\h_learning\rovers"
for i,problem in enumerate(problems):
    problem_id = problem["problem_id"]
    plan = api.get_plan(problem_id)
    if plan is None:
        continue

    file_name = "plan_for_p" + str(problem_id-1390)
    path = os.path.join(path_for_plans, file_name)
    f = open(path, "w+")
    for step in plan:
        match = re.search("^;", step)
        if match:
            continue
        f.write(step + "\n")
    f.close()

    print("problem number {} upper bound is {} lower bound is {} and the plan length {}".
          format(problem_id, problem["upper_bound"], problem["lower_bound"], len(plan)-1))
