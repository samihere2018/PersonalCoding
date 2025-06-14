#---------------------------------------------------------------------------------------------------------------------------------------------------------
#Author(s): Sylvia Amihere @ SMU
#Description: Python script to calculate the solution to system of equations using Anderson Acceleration with fixed point iteration.
#             The algorithm used is from "Performance of Low Synchronization Orthogonalization Methods in Anderson Accelerated Fixed Point Solvers"
#             by Shelby Lockhart, David J. Gardner, Carol S. Woodward, Stephen Thomas, Luke N. Olson
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# This example solves the nonlinear system
#  
# 3x - cos((y-1)z) - 1/2 = 0
# x^2 - 81(y-0.9)^2 + sin(z) + 1.06 = 0
# exp(-x(y-1)) + 20z + (10 pi - 3)/3 = 0
# 
# using the accelerated fixed pointer solver in KINSOL. The nonlinear fixed point function is
# 
#  g1(x,y,z) = 1/3 cos((y-1)yz) + 1/6
#  g2(x,y,z) = 1/9 sqrt(x^2 + sin(z) + 1.06) + 0.9
#  g3(x,y,z) = -1/20 exp(-x(y-1)) - (10 pi - 3) / 60
#
#  This system has the analytic solution x = 1/2, y = 1, z = -pi/6.
#  -------------------------------------------------------------------------------------------------

import numpy as np 

#store fixed point equations as a vector for easy substitution
def G(vec):
    x,y,z = vec.flatten() 
    eq1 = (1.0/4.0)*(np.sin(y) + complex(0,1)*z + 1.0)
    eq2 = (1.0/5.0)*(x**2 + np.cos(z) + complex(0,2))
    eq3 = (1.0/6.0)*(np.exp(-x) + y + 3.0)
    return np.array([[eq1], [eq2], [eq3]])
# end function G
    
trueSol = np.array([[complex(0.2844,0.2703)], [complex(0.1612, 0.4262)], [complex(0.6477,0.0375)]])
x0 = np.array([[complex(0.0,0.0)], [complex(0.0, 0.0)], [complex(0.0,0.0)]])
x1 = G(x0)
f0 = x1 - x0
# print("f0 =", f0)

x2 = G(x1)
f1 = x2 - x1
# print("f1 =", f1)

m        = 2
tol      = 1e-6
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
    print("Computed vector at iteration %d = \n " %i, x3)

    diff_vecs = x3 - x2
    solErr = abs(x3-trueSol)

    if np.linalg.norm(diff_vecs)<tol:
        print("~~~~~~~~~~~~~~ Solution Stats ~~~~~~~~~~~~~~~~~ ")
        print("Convergence reached at %d iteration\n" %i)
        print("~~~~~~~~~~~~~~ Exact Solution ~~~~~~~~~~~~~~~~~ ")
        print(trueSol)
        print("~~~~~~~~~~~~~~ Computed Solution ~~~~~~~~~~~~~~~~~ ")
        print(x3)
        print("~~~~~~~~~~~~~~ Solution Error~~~~~~~~~~~~~~~~~~ ")
        print(solErr)
        break
    else:
        x0 = x1
        x1 = x2
        x2 = x3
    # end if statement
# end for loop

