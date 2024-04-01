import numpy as np
import numpy_groupies as npg 
from timeit import default_timer as timer

''' Calculates median for a matrix with row groupings. 
    More performant than rows + column groupings in the group_median_rows_cols function. '''

def row_group_median(matrix:np.array, 
                     row_idx:np.array, 
                     col_idx:np.array,
                     ravel=True) -> np.array:
    
    assert matrix.ndim == 2
    assert (row_idx.size == col_idx.size == mat.size)

    no_rows = matrix.shape[0]
    no_cols = matrix.shape[1]

    matrix = matrix.ravel(order='F')
    row_idx = row_idx.ravel(order='F')
    col_idx = col_idx.ravel(order='F')

    matrix = npg.aggregate(group_idx=np.vstack([row_idx, col_idx]), a=matrix, 
                               func='sort', size=(no_rows, no_cols))
    min = npg.aggregate(group_idx=np.vstack([row_idx, col_idx]), a= matrix.reshape(-1), func='nanargmin')
    no_elements = npg.aggregate(group_idx=np.vstack([row_idx, col_idx]), a=matrix.reshape(-1), func='nanlen')

    min = min.ravel(order='F')
    no_elements = no_elements.ravel(order='F')

    # Which index the middle element is on. Two cases: 
    # 1) Odd number of elements in the group => 'middle' is not a whole number.
    #    Median is on a single row. 
    # 2) Even number of elements in the group => 'middle' is a whole number. 
    #    Median is between two rows. 

    middle = (no_elements) / 2
    middle_floor = np.floor(middle)
    middle_ceiling = np.ceil(middle)

    # Middle row...
    bottom = (middle_ceiling - 1) + min
    #... and the following row (for when the median is between two rows)
    top = (middle_floor) + min

    # If the middle point isn't between rows (middle is not a whole number), mask the top
    top[middle != middle_floor] = np.nan 

    # If all elements in a group are nan, must mask top and bottom.
    top[no_elements == 0] = np.nan 
    bottom[no_elements == 0] = np.nan 

    # Retrieve elements from index values
    bottom[~np.isnan(bottom)] = np.take(matrix, bottom[~np.isnan(bottom)].astype(int))
    top[~np.isnan(top)] = np.take(matrix, top[~np.isnan(top)].astype(int))

    # Either return result as flattened or by groupings
    if ravel == True:
        median = np.nanmean(np.vstack((bottom, top)), axis=0)
    else: 
        median_groups = np.vstack((bottom, top)).reshape(-1)
        row_idx_unique = np.unique(row_idx)
        row_id = np.tile(np.repeat(row_idx_unique, no_cols),2)
        col_id = np.tile(np.arange(0, no_cols), 2 * len(row_idx_unique))
        median = npg.aggregate(group_idx=np.vstack([row_id, col_id]), a=median_groups, func='nanmean')

    return median


# || ________ Test Example 1 ________

mat = np.array([[np.nan,     np.nan, np.nan,  np.nan,   np.nan], 
                [np.nan,     5,      np.nan,  7,        6], 
                [np.nan,     9,      np.nan,  11,       7],
                 [1,    2,      3,       np.nan,   8],
                 [5,    6,      7,       8,        9]])

row_idx = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]])

col_idx = np.array([[0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4], 
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]])


# # || ________ Test Example 2 ________

# # Create an array of custom size to test speed
# arr_size = 3000
# np.random.seed(1)
# row_idx, col_idx = np.meshgrid(np.arange(0,arr_size), np.arange(0,arr_size), indexing='ij') 
# mat = np.full(shape=(arr_size, arr_size), fill_value = np.random.randint(0, 9, size=(arr_size, arr_size)))
# row_idx = row_idx.reshape(-1)
# col_idx = col_idx.reshape(-1)
# # Make second last two rows a group 
# row_idx[-arr_size:] = row_idx[-(2*arr_size):-arr_size]

start = timer()
result = row_group_median(matrix=mat, row_idx=row_idx, col_idx=col_idx, ravel=False)
end = timer()
print('total time', end - start)



