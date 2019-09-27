import numpy as np
import math
from scipy.optimize import linear_sum_assignment

class Assignment:
    """
    source : N X 3 matrix
    destination : N X 3 matrix
    """
    def __init__(self, source_matrix = None, destination_matrix = None):
        # Save input
        self._isChange = False
        self._source_matrix = source_matrix
        self._destination_matrix = destination_matrix
        # Source < Destination
        if len(source_matrix) > len(destination_matrix):
            self._source_matrix, self._destination_matrix = self._destination_matrix, self._source_matrix
            self.isChange = True

        # Cost Matrix
        self._cost_matrix = np.zeros((len(self._source_matrix), len(self._destination_matrix)))

        # Result from algorithm
        self._source = []

        self._destination = []

    def calculate(self):
        # Set cost matrix
        for i in range(len(self._source_matrix)):
            for j in range(len(self._destination_matrix)):
                for k in range(len(self._destination_matrix[0])):
                    self._cost_matrix[i][j] += (self._destination_matrix[j][k] - self._source_matrix[i][k]) ** 2
                self._cost_matrix[i][j] = math.sqrt(self._cost_matrix[i][j])
        
        # Calulate Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(self._cost_matrix)

        # Save result
        print("source :", len(self._source))
        print("destination:", len(self._destination))
        self._source += self._source_matrix[row_ind].tolist()
        self._destination += self._destination_matrix[col_ind].tolist()

        # Check if do more calculate
        print("source_matrix len :", len(self._source_matrix))
        print("destination_matrix len:", len(self._destination_matrix))
        if len(self._source_matrix) < len(self._destination_matrix):
            new_destination_matrix = []
            exclude = set(col_ind)

            for idx in range(len(self._destination_matrix)):
                if not idx in exclude:
                    new_destination_matrix.append(self._destination_matrix[idx])
            self._destination_matrix = np.array(new_destination_matrix)
            self._cost_matrix = np.zeros((len(self._source_matrix), len(self._destination_matrix)))

            # Reculsive
            self.calculate()
        
    # Get results after calculation
    def get_result(self):
        if self._isChange is True:
            return self._destination, self._source
        else :
            return self._source, self._destination

    