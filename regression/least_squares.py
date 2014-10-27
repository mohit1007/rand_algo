from factorization.tsqr import *
from rand_solvers.projections import *

class LeastSquares:
    def __init__(self, matrix, sc):
        self.matrix = matrix
        self.sc = sc

    def compute_leverage_scores(self, num_projections, c):
        projection_matrix = self.__project(num_projections, c)
        R = self.__factorize(projection_matrix)
        lev = self.matrix.get_rdd().map(lambda row: np.linalg.norm(np.dot(row, R))**2)
        num_cols = self.matrix.get_dimensions()[1]
        sum_vel = num_cols*np.array(lev.collect())/lev.reduce(add)
        app_lev = np.array(sum_vel)
        return app_lev

    def __project(self, num_projections, c):
        projection = Projections(self.matrix, self.sc)
        projection_matrix = projection.SpaFJLT(num_projections, c)
        return projection_matrix

    def __factorize(self, projection_matrix):
        tsqr = TSQR(projection_matrix, 3, self.sc)
        R = tsqr.tsqr()
        R = np.linalg.inv(R)
        return R

