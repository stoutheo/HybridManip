####################
# Extrenal libs
####################
# casadi
from py_pack import ca


# build basic operators to be used


################################################################################
#--------------------------- Rotation Martix ----------------------------------#
################################################################################
# init symbolic variable for the angle of the Rot mat
param_fi = ca.SX.sym("param_fi")

# init symbolic variable for Rot mat
R_mat = ca.SX.sym("R_mat",2,2)
R_mat[0,0] = ca.cos(param_fi)
R_mat[0,1] = -ca.sin(param_fi)
R_mat[1,0] = ca.sin(param_fi)
R_mat[1,1] = ca.cos(param_fi)

# 2D rotation matrix
R = ca.Function('R',[param_fi], [R_mat])
#--------------------------- Rotation Martix End ------------------------------#

################################################################################
#--------------------- Parametric equation of Circle --------------------------#
################################################################################

# init symbolic variable for the angle of the point w.r.t the circle
param_theta = ca.SX.sym("param_theta")
# init symbolic variable for the radius of the circle
param_radius = ca.SX.sym("param_radius")

# define circle parametric equation
pointOnCircle = ca.vertcat(param_radius*ca.cos(param_theta),
                           param_radius*ca.sin(param_theta));

# point on a circle given the radius and the angle
circlePt = ca.Function('circlePt',[param_radius, param_theta], [pointOnCircle])

#-------------------- Parametric equation of Circle End------------------------#
