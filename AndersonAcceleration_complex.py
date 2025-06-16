#---------------------------------------------------------------------------------------------------------------------------------------------------------
#Author(s): Sylvia Amihere @ SMU
#Description: Python script to calculate the solution to system of equations using Anderson Acceleration with fixed point iteration.
#             The algorithm used is from "Performance of Low Synchronization Orthogonalization Methods in Anderson Accelerated Fixed Point Solvers"
#             by Shelby Lockhart, David J. Gardner, Carol S. Woodward, Stephen Thomas, Luke N. Olson
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# This example solves the nonlinear system
#  
# 4x    - sin(y) - zi     - 1  = 0
# -x^2  + 5y     - cos(z) - 2i = 0
# -e^-x -y      + 6z      - 3  = 0
# 
# using the accelerated fixed pointer solver in KINSOL. The nonlinear fixed point function is
# 
#  g1(x,y,z) = (1/4)*(sin(y) + zi + 1)
#  g2(x,y,z) = (1/5)*(x^2 + cos(z) + 2i)
#  g3(x,y,z) = (1/6)*(exp(-x) + y + 3)
#
#  This system has the analytic solution: 
#                                        x = 0.2844 + 0.2703i 
#                                        y = 0.1612 + 0.4262i
#                                        z = 0.6477 + 0.0375i
#  -------------------------------------------------------------------------------------------------

import numpy as np 

#store fixed point equations as a vector for easy substitution
def G(vec):
    x,y,z = vec.flatten() 
    eq1 = (1.0/4.0)*(np.sin(y) + complex(0.0,1.0)*z + 1.0)
    eq2 = (1.0/5.0)*(x**2 + np.cos(z) + complex(0.0,2.0))
    eq3 = (1.0/6.0)*(np.exp(-x) + y + 3.0)
    return np.array([[eq1], [eq2], [eq3]])
# end function G
    
#exact solution
trueSol = np.array([[complex(0.28443101049565, 0.27031686078054)], [complex(0.16117132843381, 0.42622240595676)], [complex(0.64771494226506, 0.03754877135588)]])

#initial guess
x0 = np.array([[complex(0.2,0.2)], [complex(0.1, 0.4)], [complex(0.6,0.03)]])
x1 = G(x0)
f0 = x1 - x0
# print("f0 =", f0)

x2 = G(x1)
f1 = x2 - x1
# print("f1 =", f1)

m        = 2
tol      = 1.49012e-06
maxIters = 30

for i in range(2,  maxIters):
    f2 = G(x2) - x2

    delta_g0 = G(x1) - G(x0)
    delta_g1 = G(x2) - G(x1)
    big_G = np.column_stack((delta_g0, delta_g1))

    delta_f0 = (G(x1) - x1) - (G(x0) - x0)#f1 - f0
    delta_f1 = (G(x2) - x2) - (G(x1) - x1)#f2 - f1
    big_F = np.column_stack((delta_f0, delta_f1))

    Q, R = np.linalg.qr(big_F)
    ls_gamma = np.linalg.solve(R, Q.T @ f2)

    bigG_gamma = np.matmul(big_G, ls_gamma)
    x3 = G(x2) - bigG_gamma
    print("Computed vector at iteration %d = \n " %i)
    for element in x3:
        print(f"{element[0]:.15E}")
    print(" \n ")

    diff_vecs = x3 - x2
    solErr = abs(x3-trueSol)

    if np.linalg.norm(diff_vecs)<tol:
        print("~~~~~~~~~~~~~~ Solution Stats ~~~~~~~~~~~~~~~~~ ")
        print("Convergence reached at iteration %d\n" %i)
        print("~~~~~~~~~~~~~~ Exact Solution ~~~~~~~~~~~~~~~~~ ")
        for element in trueSol:
            print(f"{element[0]:.6E}")
        # print(trueSol)
        print("~~~~~~~~~~~~~~ Computed Solution ~~~~~~~~~~~~~~~~~ ")
        for element in x3:
            print(f"{element[0]:.6E}")
        print("~~~~~~~~~~~~~~ Solution Error~~~~~~~~~~~~~~~~~~ ")
        for element in solErr:
            print(f"{element[0]:.6E}")
        # print(solErr)
        break
    else:
        x0 = x1
        x1 = x2
        x2 = x3
    # end if statement
# end for loop

