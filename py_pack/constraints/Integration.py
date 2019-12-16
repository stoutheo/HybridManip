#-----------------------------------------------------------------------------#
# This file has basic functions to enforces the integration constraints
# .....
# .....
#-----------------------------------------------------------------------------#

####################
# Extrenal libs
####################
# casadi
from py_pack import ca
# numpy
from py_pack import np


### -------------------------------------------------------------------------###
###                         Integration equation                             ###
### -------------------------------------------------------------------------###

# ---------------------------- trapezoidal quadrature -------------------------#
integr = lambda v, u, dv, du, dt : (u - v -(0.5)*(du+dv)*dt) # integrating pos->vel

# --------------------- trapezoidal quadrature dynamics -----------------------#
dynam = lambda v, u, dv, du, dt, g, fext_v, fext_u : (u - v -(0.5)*(du+dv+2*g+fext_v+fext_u)*dt) # integrating force->vel


### -------------------------------------------------------------------------###
###                              Time variable                               ###
### -------------------------------------------------------------------------###

# delta time between know k and k+1
param_dt = ca.SX.sym("param_dt")

### -------------------------------------------------------------------------###
###                              1D vectors                                  ###
### -------------------------------------------------------------------------###

n=1
# -------------- quantities at current knot
# state related params (x and y coords) at current knot (k)
param_state_k = ca.SX.sym("param_state_k"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at current knot (k)
param_dstate_k = ca.SX.sym("param_dstate_k"+"_"+str(n)+"D", n,1)
# external force at current knot (k)
param_extF_k = ca.SX.sym("param_extF_k"+"_"+str(n)+"D", n,1)

# -------------- quantities at next knot
# state related params (x and y coords) at next knot (k+1)
param_state_k1 = ca.SX.sym("param_state_k1"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at next knot (k+1)
param_dstate_k1 = ca.SX.sym("param_dstate_k1"+"_"+str(n)+"D", n,1)
# external force at next knot (k+1)
param_extF_k1 = ca.SX.sym("param_extF_k1"+"_"+str(n)+"D", n,1)

# -------------- general quantities
G = ca.SX.sym("gravity", n,1)

# integration with trapezoidal quadrature for 1D
trap_quad = ca.Function('trap_quad',\
               [param_state_k, param_state_k1, param_dstate_k, param_dstate_k1, param_dt],\
               [integr(param_state_k, param_state_k1, param_dstate_k, param_dstate_k1, param_dt)])


# dynamis with trapezoidal quadrature for 1D
dynam_tq = ca.Function('dynam_trap_quad',\
               [param_state_k, param_state_k1, param_dstate_k, param_dstate_k1, param_dt, G, param_extF_k, param_extF_k1 ],\
               [dynam(param_state_k, param_state_k1, param_dstate_k, param_dstate_k1, param_dt, G, param_extF_k, param_extF_k1)])


### -------------------------------------------------------------------------###
###                              2D vectors                                  ###
### -------------------------------------------------------------------------###
n=2
# -------------- quantities at current knot
# state related params (x and y coords) at current knot (k)
param_state_k_2D = ca.SX.sym("param_state_k"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at current knot (k)
param_dstate_k_2D = ca.SX.sym("param_dstate_k"+"_"+str(n)+"D", n,1)
# external force at current knot (k)
param_extF_k_2D = ca.SX.sym("param_extF_k"+"_"+str(n)+"D", n,1)

# -------------- quantities at next knot
# state related params (x and y coords) at next knot (k+1)
param_state_k1_2D = ca.SX.sym("param_state_k1"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at next knot (k+1)
param_dstate_k1_2D = ca.SX.sym("param_dstate_k1"+"_"+str(n)+"D", n,1)
# external force at next knot (k+1)
param_extF_k1_2D = ca.SX.sym("param_extF_k1"+"_"+str(n)+"D", n,1)

# -------------- general quantities
G_2D = ca.SX.sym("gravity", n,1)

# integration with trapezoidal quadrature for 2D using the elementary function (trap_quad))
trap_quad2D = ca.Function('trap_quad2D',\
                 [param_state_k_2D, param_dstate_k_2D, param_state_k1_2D, param_dstate_k1_2D, param_dt],\
                 [ca.vertcat(\
                 trap_quad(param_state_k_2D[0], param_dstate_k_2D[0], param_state_k1_2D[0], param_dstate_k1_2D[0], param_dt),\
                 trap_quad(param_state_k_2D[1], param_dstate_k_2D[1], param_state_k1_2D[1], param_dstate_k1_2D[1], param_dt))])

# dynamis with trapezoidal quadrature for 2D
dynam_tq2D = ca.Function('dynam_trap_quad2D',\
               [param_state_k_2D, param_state_k1_2D, param_dstate_k_2D, param_dstate_k1_2D, param_dt, G_2D, param_extF_k_2D, param_extF_k1_2D ],\
                [ca.vertcat(\
                dynam(param_state_k_2D[0], param_state_k1_2D[0], param_dstate_k_2D[0], param_dstate_k1_2D[0], param_dt, G_2D[0], param_extF_k_2D[0], param_extF_k1_2D[0]),\
                dynam(param_state_k_2D[1], param_state_k1_2D[1], param_dstate_k_2D[1], param_dstate_k1_2D[1], param_dt, G_2D[1], param_extF_k_2D[1], param_extF_k1_2D[1]))])

### -------------------------------------------------------------------------###
###                              3D vectors                                  ###
### -------------------------------------------------------------------------###
n=3
# -------------- quantities at current knot
# state related params (x and y coords) at current knot (k)
param_state_k_3D = ca.SX.sym("param_state_k"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at current knot (k)
param_dstate_k_3D = ca.SX.sym("param_dstate_k"+"_"+str(n)+"D", n,1)
# external force at current knot (k)
param_extF_k_3D = ca.SX.sym("param_extF_k"+"_"+str(n)+"D", n,1)
# -------------- quantities at next knot
# state related params (x and y coords) at next knot (k+1)
param_state_k1_3D = ca.SX.sym("param_state_k1"+"_"+str(n)+"D", n,1)
# derivative of state related params (dx/dt and dy/dt coords) at next knot (k+1)
param_dstate_k1_3D = ca.SX.sym("param_dstate_k1"+"_"+str(n)+"D", n,1)
# external force at next knot (k+1)
param_extF_k1_3D = ca.SX.sym("param_extF_k1"+"_"+str(n)+"D", n,1)

# -------------- general quantities
G_3D = ca.SX.sym("gravity", n,1)

# integration with trapezoidal quadrature for 2D using the elementary function (trap_quad))
trap_quad3D = ca.Function('trap_quad3D',\
                 [param_state_k_3D, param_dstate_k_3D, param_state_k1_3D, param_dstate_k1_3D, param_dt],\
                 [ca.vertcat(\
                 trap_quad(param_state_k_3D[0], param_dstate_k_3D[0], param_state_k1_3D[0], param_dstate_k1_3D[0], param_dt),\
                 trap_quad(param_state_k_3D[1], param_dstate_k_3D[1], param_state_k1_3D[1], param_dstate_k1_3D[1], param_dt),\
                 trap_quad(param_state_k_3D[2], param_dstate_k_3D[2], param_state_k1_3D[2], param_dstate_k1_3D[2], param_dt))])


# dynamis with trapezoidal quadrature for 2D
dynam_tq3D = ca.Function('dynam_trap_quad3D',\
               [param_state_k_3D, param_state_k1_3D, param_dstate_k_3D, param_dstate_k1_3D, param_dt, G_3D, param_extF_k_3D, param_extF_k1_3D ],\
                [ca.vertcat(\
                dynam(param_state_k_3D[0], param_state_k1_3D[0], param_dstate_k_3D[0], param_dstate_k1_3D[0], param_dt, G_3D[0], param_extF_k_3D[0], param_extF_k1_3D[0]),\
                dynam(param_state_k_3D[1], param_state_k1_3D[1], param_dstate_k_3D[1], param_dstate_k1_3D[1], param_dt, G_3D[1], param_extF_k_3D[1], param_extF_k1_3D[1]),\
                dynam(param_state_k_3D[2], param_state_k1_3D[2], param_dstate_k_3D[2], param_dstate_k1_3D[2], param_dt, G_3D[2], param_extF_k_3D[2], param_extF_k1_3D[2]))])


# end of file
