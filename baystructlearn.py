############################################################
#               Bayesian Structure Learning                #
#                     By: Aaron Jin                        #
############################################################

import sys
import pandas as pd
import networkx as nx
from itertools import permutations
from math import lgamma
import random


# Step 1: Data preprocessing
def read_data(filename):
    data = pd.read_csv(filename)
    return data


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


# Step 2: Scoring function
def compute_bayesian_score(data, dag, alpha):
    score = 0.0
    vars = data.columns

    for var in vars:
        parents = list(dag.predecessors(var))
        var_data = data[var]

        if parents:
            parent_data = data[parents]
            counts = parent_data.groupby(parents).size()
            unique_configs = counts.index

            for config in unique_configs:
                if not isinstance(config, tuple):
                    config = (config,)
                
                subset = var_data[parent_data.apply(lambda row: tuple(row) == config, axis = 1)]
                m_ij0 = len(subset)
                score += lgamma(alpha[var][0]) - lgamma(alpha[var][0] + m_ij0)
                value_counts = subset.value_counts()

                for state, count in value_counts.items():
                    score += lgamma(alpha[var][state] + count) - lgamma(alpha[var][state])

        else:
            m_ij0 = len(var_data)
            score += lgamma(alpha[var][0]) - lgamma(alpha[var][0] + m_ij0)
            value_counts = var_data.value_counts()

            for state, count in value_counts.items():
                score += lgamma(alpha[var][state] + count) - lgamma(alpha[var][state])
    
    return score


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
