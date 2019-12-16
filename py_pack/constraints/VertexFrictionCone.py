#-----------------------------------------------------------------------------#
# This file has basic functions to compute the constraint that enforces that
# the force has to lie with the friction cone.
# The actual equation used is the vertex form of the friction cone :
#  F = a*lV + b*rV , where a,b are scalar coefficients,
#                           lV, rV are the basis vectors of the friction Cone
#                           in other words the edges of the linearized cone
#
# relevant reference for the linearized friction cone in the traditional form
# (half space representation or polytope) and not in the vertex can be found
# here: https://scaron.info/teaching/friction-cones.html
# regarding the different representations relevant info can be found
# here:  https://scaron.info/teaching/polyhedra-and-polytopes.html
#-----------------------------------------------------------------------------#

####################
# Extrenal libs
####################
# casadi
from py_pack import ca
# numpy
from py_pack import np

####################
# Custom package libs
####################
# info of objects
from py_pack import obj_Dist_Norm_Cone
#-----------------------------------------------------------------------------#

# init general params with respect to the dist constraint
param_or = ca.SX.sym("param_or")     # polar-coord angle
# init general params with respect to the normal constraint
param_fiObj = ca.SX.sym("param_fiObj")   # global angle (orientation) of object

# init general params with respect to the friction cone
param_muFric = ca.SX.sym("param_muFric")   # angle to normal of friction cone

# init general params of the scalar coefficients
param_ForceCoef = ca.SX.sym("param_ForceCoef", 2)

# compute the 2D force vector allowed inside the friction cone manifold
comp_force = param_ForceCoef[0]*obj_Dist_Norm_Cone.rVertexCone(param_or, param_fiObj, param_muFric)\
           + param_ForceCoef[1]*obj_Dist_Norm_Cone.lVertexCone(param_or, param_fiObj, param_muFric)

#  obtain 2D force vector allowed inside the friction cone manifold
param_force = ca.Function('param_force',[param_or, param_fiObj, param_muFric, param_ForceCoef], \
              [comp_force])
