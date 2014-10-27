from pyspark import SparkContext
from regression.least_squares import *

if __name__ == "__main__":
    sc = SparkContext()
    n = 52768
    d = 50
    c = 1
    num_projections = 200
    matrix_list, matrix = gen_synth_data(n, d)
    rdd = sc.parallelize(matrix_list)

    spark_matrix = Matrix(rdd)
    ls = LeastSquares(spark_matrix, sc)
    app_lev = ls.compute_leverage_scores(num_projections, c)


    U, D, V = np.linalg.svd(matrix, full_matrices=False)
    lev_exact = np.sum(U**2, axis=1)
    error = np.linalg.norm(lev_exact - app_lev) / np.linalg.norm(lev_exact)
    print "error ", error


