__author__ = 'msingh'
from regression.least_squares import *
from pyspark import SparkContext
from utils.matrix import *
from collections import defaultdict

def ideal_cost(A, b):
    x_opt = npl.lstsq(A, b)[0]
    f_opt = npl.norm(np.dot(A, x_opt)-b)
    return x_opt, f_opt


if __name__ == "__main__":

    sc = SparkContext(appName="test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')

    A = Ab[:, :-1]
    b = Ab[:, -1]
    x_opt, f_opt = ideal_cost(A, b)
    A_rdd = sc.parallelize(A.tolist()) # Create a RDD for the rows of A.
    b_rdd = sc.parallelize(b)

    matrix_A = Matrix(A_rdd)
    matrix_b = Matrix(b_rdd)
    k_param = 20
    global_costs = defaultdict(dict)
    for technique in ['projection', 'sampling']:
        for method in ['srdht', 'cw', 'gaussian', 'rademacher']:
            ls = RandLeastSquares(matrix_A, matrix_b, sc=sc, k=k_param, c=3000, s=2000, method=method, technique=technique)
            x, costs = ls.fit(A, b)
            scaled_weights_error = []
            scaled_cost_error = []
            for i in range(k_param):
                w_err = npl.norm(np.array(x[i])-x_opt)/npl.norm(x_opt)
                c_err = np.abs(f_opt - ls.cost[i])/f_opt
                logger.info("scaled weights error " + str(w_err))
                logger.info("scaled cost error "+str(c_err))
                scaled_weights_error.append(w_err)
                scaled_cost_error.append(c_err)
            global_costs[technique][method] = {'absolute_costs': ls.cost, 'relative costs': scaled_cost_error, 'relative weights': scaled_weights_error}
    print global_costs

