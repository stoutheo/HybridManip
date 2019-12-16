#-----------------------------------------------------------------------------#
# This file has basic functions to compute the constraint that computes the
# torque w.r.t the respective force at the contact location.
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
# package utils
from py_pack import operators as oper
#-----------------------------------------------------------------------------#

# force and torque related params
param_force = ca.SX.sym("param_force", 2)
param_torque_Cirl = ca.SX.sym("param_torque_Cirl")
param_torque_Obj = ca.SX.sym("param_torque_Obj")

# inertia parameter
param_Inertia = ca.SX.sym("param_Inertia")
# mass parameter
param_mass = ca.SX.sym("param_mass")

# contact angle
param_or = ca.SX.sym("param_or")  # polar-coord angle
# contact distance
param_radius = ca.SX.sym("param_radius")   # polar-coord radius
# init general params with respect to the normal constraint
param_fiObj = ca.SX.sym("param_fiObj")     # global angle (orientation) of object

#-----------------------------------------------------
# version 1

# use of the circle parametric equation to obtain the coords of the contact point
# obtain the contact point after the rotation of the body has been included
c_p_Cirl = oper.circlePt(param_radius, param_or + param_fiObj)

# compute the torque
comp_torque_Cirl = ((c_p_Cirl[0]*param_force[1] - c_p_Cirl[1]*param_force[0])*param_mass)/param_Inertia

#  obtain torque from force vector according to the contact location and
param_torque_Cirl = ca.Function('param_torque_Cirl',\
               [param_or, param_fiObj, param_radius, param_Inertia, param_mass, param_force],\
               [comp_torque_Cirl])

#-----------------------------------------------------
# version 2

# use of the object function + rotation to obtain the coords of the contact point
# rotate the contact point given the rotation of the body
c_p_Obj = ca.mtimes(oper.R(param_fiObj), obj_Dist_Norm_Cone.xy_spl(param_or).T)

# compute the torque
comp_torque_Obj = ((c_p_Obj[0]*param_force[1] - c_p_Obj[1]*param_force[0])*param_mass)/param_Inertia

#  obtain torque from force vector according to the contact location and
param_torque_Obj = ca.Function('param_torque_Obj',\
               [param_or, param_fiObj, param_Inertia, param_mass, param_force],\
               [comp_torque_Obj])

#
