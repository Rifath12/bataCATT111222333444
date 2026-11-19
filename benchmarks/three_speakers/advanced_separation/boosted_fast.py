"""Optional single-threaded evaluator for the fixed close-microphone regression trees."""
import numpy as np
from numba import njit

@njit
def evaluate(features, columns, thresholds, left, right, values, intercept):
    result = np.full(features.shape[0], intercept, dtype=np.float64)
    for tree in range(columns.shape[0]):
        for sample in range(features.shape[0]):
            node = 0
            while columns[tree, node] >= 0:
                column = columns[tree, node]
                if features[sample, column] <= thresholds[tree, node]:
                    node = left[tree, node]
                else:
                    node = right[tree, node]
            result[sample] += values[tree, node]
    return result

def compile_predictor(trees, intercept):
    width = max(len(tree.value) for tree in trees)
    shape = (len(trees),width)
    arrays = [np.zeros(shape,dtype=t) for t in (np.int32,np.float64,np.int32,np.int32,np.float64)]
    for row,tree in enumerate(trees):
        for a,v in zip(arrays,(tree.column,tree.threshold,tree.left,tree.right,tree.value),strict=True):
            a[row,:len(v)] = v
    columns, thresholds, left, right, values = arrays

    def predict(features):
        return evaluate(features, columns, thresholds, left, right, values, intercept)

    return predict
