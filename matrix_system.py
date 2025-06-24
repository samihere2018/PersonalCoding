#---------------------------------------------------------------------------------------------------------
#Author(s): Sylvia Amihere @ SMU
#Description: The example below solves a real-valued and complex valued matrix-system for the solution x.
#---------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# This example solves the real-valued and complex-valued matrix system
# Real-valued:    A = 
#                 b = 
# Complex-valued: A = 
#                 b = 
#  -------------------------------------------------------------------------------------------------------

import numpy as np

# realValued_matrixVector = True
realValued_matrixSystem = True

# complexValued_matrixVector = True
complexValued_matrixSystem = True


# if (realValued_matrixVector):
#     # real-valued matrix-vector multiplication
#     mat_A = np.array([[7.0, 8.0, 7.0, 10.0, 4.0], 
#                       [5.0, 1.0, 6.0, 8.0, 10.0], 
#                       [8.0, 6.0, 2.0, 1.0, 5.0],
#                       [4.0, 9.0, 7.0, 10.0, 3.0], 
#                       [6.0, 4.0, 2.0, 8.0, 5.0]])
#     vec_x = np.array([[3.0], 
#                       [1.0], 
#                       [2.0], 
#                       [4.0], 
#                       [8.0]])
#     vec_b = mat_A.dot(vec_x)
#     print("Solution for matrix-vector multiplication:", vec_b)

if (realValued_matrixSystem):
    # real-valued matrix-vector multiplication
    mat_A1 = np.array([[7.0, 8.0, 7.0, 10.0, 4.0], 
                       [5.0, 1.0, 6.0, 8.0, 10.0], 
                       [8.0, 6.0, 2.0, 1.0, 5.0],
                       [4.0, 9.0, 7.0, 10.0, 3.0], 
                       [6.0, 4.0, 2.0, 8.0, 5.0]])
    vec_b1 = np.array([[115.0], 
                       [140.0], 
                       [78.0], 
                       [99.0], 
                       [98.0]])
    vec_x1 = np.linalg.solve(mat_A1, vec_b1)
    print("Solution to matrix-system:", vec_x1)


    if (complexValued_matrixSystem):
        # complex-valued matrix-vector multiplication
        mat_AC = np.array([[complex(10.0, 6.0), complex(9.0, 4.0), complex(3.0, 2.0), complex(8.0, -6.0), complex(7.0, 4.0)], 
                           [complex(7.0, 3.0), complex(8.0, -2.0), complex(10.0, 3.0), complex(5.0, 4.0), complex(2.0, -1.0)], 
                           [complex(3.0, -1.0), complex(1.0, 1.0), complex(1.0, -2.0), complex(6.0, 1.0), complex(4.0, 3.0)], 
                           [complex(8.0, 7.0), complex(7.0, 7.0), complex(6.0, -4.0), complex(2.0, 3.0), complex(9.0, -3.0)], 
                           [complex(5.0, 2.0), complex(4.0, -4.0), complex(9.0, 1.0), complex(1.0, 5.0), complex(4.0, 1.0)]])
        vec_bC = np.array([[complex(2.0, 3.0)],
                           [complex(3.0, 4.0)],
                           [complex(8.0, 5.0)],
                           [complex(2.0, -1.0)],
                           [complex(1.0, 1.0)]]) 
        vec_xC = np.linalg.solve(mat_AC, vec_bC)
        print("Solution to matrix-system:", vec_xC)


