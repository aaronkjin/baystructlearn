# BayStructLearn

A structure learning approach to find best-fit Bayesian networks using K2 search algorithm. A CS 238 project.

## Getting Started

```bash
# clone the repo
git clone https://github.com/aaronkjin/baystructlearn.git
cd baystructlearn

# create a virtual env
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

To run the script:

```bash
python3 baystructlearn.py data/<input_file>.csv output/<output_file>.gph
```

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- NetworkX

## Background

In this project, I focus on Bayesian network structure learning, which involves discovering the optimal network structure that best represents the dependencies in a given dataset. Here, I use the K2 algorithm, a greedy, heuristic search approach, where this implementation efficiently constructs a DAG by iteratively adding parent nodes to each variable to maximize the Bayesian score. I also implement optimizations using vectorized operations with Pandas and NumPy to reduce execution time and ensure performance, since our datasets have varying sizes.

## Developer

Aaron Jin  
[GitHub Profile](https://github.com/aaronkjin)
