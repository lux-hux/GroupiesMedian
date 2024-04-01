import numpy as np
import numpy_groupies as npg
from timeit import default_timer as timer

''' Calculates median for a matrix with row and column groupings 
    Less performant than only row or column groupings in the group_median_rows function. '''

def group_median_rows_cols(matrix:np.array, 
                          row_idx:np.array, 
                          col_idx:np.array) -> np.array:

    # !Largest performance drag!:
    # Return array containing sub-arrays of elements from each group. 
    # Edge-case: if all groups contain one element (no grouping), the function below will return a list
    matrix = npg.aggregate(group_idx=np.vstack([row_idx, col_idx]), a=matrix.reshape(-1), func='array')
    matrix = matrix.reshape(-1)
    
    # Create an index from 0 to N for each group, where N is the number of elements in the group
    # https://stackoverflow.com/questions/70165387/store-indexes-after-concatenating-a-numpy-array
    sub_array_lens = list(map(len, matrix))
    group_ranges = np.repeat(np.arange(len(matrix)), sub_array_lens)

    matrix = np.concatenate(matrix)
    matrix = npg.aggregate(group_idx=group_ranges, a=matrix, func='sort')

    indices = np.where(group_ranges[:-1] != group_ranges[1:])[0] + 1
    indices = np.concatenate([indices, [len(group_ranges)]])

    indices = np.concatenate([[0], indices])

    middle_frac = (indices[:-1] + indices[1:]) / 2

    middle_indices_top = (indices[:-1] + indices[1:]) // 2
    middle_indices_bottom = middle_indices_top - 1

    middle_indices_bottom[middle_frac!=middle_indices_top] = -1

    median_groups = np.vstack([middle_indices_bottom, middle_indices_top])

    xs, ys = np.indices(median_groups.shape)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    median_groups_flat = median_groups.reshape(-1)
    mask = (median_groups_flat != -1)
    xs = xs[mask]
    ys = ys[mask]
    median_groups_flat = median_groups_flat[mask]
    result = np.full(shape=median_groups.shape, fill_value=np.nan)
    result[xs, ys] = matrix.take(median_groups_flat.astype(int))

    return np.nanmean(result,axis=0)

# || ________ Test Example 1 ________

matrix = np.array([[np.nan,     9,       2,  np.nan], 
                   [0,     5,      6,  np.nan], 
                   [9,     9,     10, 11]])

row_idx = np.array([0, 0, 0, 0,
                    1, 1, 1, 1,
                    2, 2, 2, 2])

col_idx = np.array([0, 1, 2, 2,
                    0, 1, 2, 2,
                    0, 1, 2, 2])

# || ________ Test Example 1 ________

# arr_size = 3000
# np.random.seed(1)
# row_idx, col_idx = np.meshgrid(np.arange(0,arr_size), np.arange(0,arr_size), indexing='ij') 
# matrix = np.full(shape=(arr_size, arr_size), fill_value = np.random.randint(0, 9, size=(arr_size, arr_size)))
# row_idx = row_idx.reshape(-1)
# col_idx = col_idx.reshape(-1)
# row_idx[-arr_size:] = row_idx[-(2*arr_size):-arr_size]

start = timer()
result = group_median_rows_cols(row_idx= row_idx, col_idx=col_idx, matrix=matrix)
end = timer()
print('total time', end - start)