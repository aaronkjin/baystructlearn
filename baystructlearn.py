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
        grouped = data.groupby(parents)[var].value_counts().unstack(fill_value = 0)
        counts_parent = grouped.sum(axis = 1).values
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
