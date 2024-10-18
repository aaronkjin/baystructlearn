############################################################
#               Bayesian Structure Learning                #
#                     By: Aaron Jin                        #
############################################################

import sys
import pandas as pd
import networkx as nx
import numpy as np
from math import lgamma
from itertools import combinations


# Step 1: Data preprocessing
def read_data(filename):
    data = pd.read_csv(filename)
    return data


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def initialize_alpha(data):
    alpha = {}

    # Using uniform priors: alpha = 1 for each state
    for var in data.columns:
        num_states = data[var].nunique()
        alpha[var] = np.ones(num_states)
    
    return alpha


# Step 2: Scoring function
def compute_variable_score(data, var, parents, alpha):
    # Edge case: No parents (score only depends on var itself)
    if not parents:
        counts = data[var].value_counts().values
        score = lgamma(alpha[var].sum()) - lgamma(alpha[var].sum() + len(data))
        score += np.sum(lgamma(alpha[var] + counts) - lgamma(alpha[var]))
        
        return score
    
    # General case: With parents (compute scores for each parent config)
    else:
        grouped = data.groupby(parents)[var].value_counts().unstack(fill_value=0)
        counts_parent = grouped.sum(axis=1).values
        counts_child = grouped.values
        alpha_parent = alpha[var][grouped.columns]

        score = lgamma(alpha[var][0]) - lgamma(alpha[var][0] + counts_parent)
        score += np.sum(lgamma(counts_child + alpha[var]) - lgamma(alpha[var]))

        return np.sum(score)


def compute_bayesian_score(data, dag, alpha, variable_order):
    total_score = 0.0

    for var in variable_order:
        parents = list(dag.predecessors(var))
        score = compute_variable_score(data, var, parents, alpha)
        total_score += score

    return total_score


# Step 3: Search algorithm
def k2_search(data, alpha, max_parents=5):
    variables = list(data.columns)
    variable_order = variables.copy()
    dag = nx.DiGraph()
    dag.add_nodes_from(variables)
    total_score = 0.0

    unique_states = {var: data[var].unique() for var in variables}

    for i, var in enumerate(variable_order):
        cur_parents = []
        best_score = compute_variable_score(data, var, cur_parents, alpha)
        improved = True

        while improved and len(cur_parents) < max_parents:
            improved = False
            candidate_parents = [v for v in variable_order[:i] if v not in cur_parents]
            scores = []

            if not candidate_parents:
                break

            # Calculate vectorized computation of scores for all candidate parents
            candidate_scores = []
            for candidate in candidate_parents:
                temp_parents = cur_parents + [candidate]
                score = compute_variable_score(data, var, temp_parents, alpha)
                candidate_scores.append((candidate, score))

            # Find candidate with highest score
            if candidate_scores:
                best_candidate, best_candidate_score = max(candidate_scores, key=lambda x: x[1])

                if best_candidate_score > best_score:
                    cur_parents.append(best_candidate)
                    best_score = best_candidate_score
                    improved = True
                else:
                    improved = False

        # Add selected parents to DAG
        for parent in cur_parents:
            dag.add_edge(parent, var)
        
        total_score += best_score
        print(f"Variable '{var}': Parents added: {cur_parents}, Score: {best_score}")
    
    return dag, total_score


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
