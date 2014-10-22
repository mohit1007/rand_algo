__author__ = 'msingh'

from utils import *


class Matrix(object):
    def __init__(self, rdd):
        self.matrix = rdd

    def transpose(self):
        rows_each_part = self.matrix.mapPartitions(lambda x: num_rows_each_partition(x)).collect()
        indexed_matrix = self.matrix.mapPartitionsWithSplit(lambda index, it: indexing(index, it, rows_each_part)). \
            flatMap(lambda x: flip(x))
        grouped = indexed_matrix.groupBy(lambda x: x[0]).map(lambda x: [e for e in x[1]])
        return Matrix(grouped.map(lambda x: extract(x)))


    def dot(self, other):
        # join by columns and sum the corresponding eles
        return Matrix(self.matrix.join(other.matrix).map(lambda (k, v): v). \
                      mapPartitions(matrixsum_iterator_other).sum())

    def take(self, rows):
        return self.matrix.matrix.take(rows)


    def top(self):
        return self.matrix.matrix.first()

    def collect(self):
        return self.matrix.collect()
