#-----------------------------------------------------------------------------#
# This part of the file has basic functions to compute the distance to
# shape constraint.
# It basically:
#    i) project the angle of a polar point to 0-360 new_range
#   ii) finds the value of that angle in the range of the spline
#  iii) computes the x and y, of the angle on the spline
#   iv) computes the Euclidean distance of x,y to the centre of the object
#    v) subtracts from this distace the distance of the polar point
#
# It utilise the objects module to obtain the spline that describes
# the surface of the object.
# Further, it assumes that the free parameters are polar-coordinates
# of a point w.r.t the centre of the object.
#-----------------------------------------------------------------------------#


####################
# Extrenal libs
####################
# casadi
from py_pack import ca
# numpy
from py_pack import np

####################
# Custom local libs
####################
# object splines
from py_pack import obj_spl

####################
# Custom package libs
####################
# package utils
from py_pack import operators as oper
#-----------------------------------------------------------------------------#

# towards build the shape constraint
# obtain the range of the spline
new_range = (obj_spl.spl_brk[-1]- obj_spl.spl_brk[0])

# init general params with respect to the dist constraint
param_or = ca.SX.sym("param_or")     # polar-coord angle
param_d = ca.SX.sym("param_d")       # polar-coord distance

# parameter of the spline from angle (polar coordinate)
theta_spl = ca.Function('theta_spl',[param_or], \
            [new_range*((param_or) - (2*np.pi)*ca.floor((param_or)/(2*np.pi)))/(2*np.pi)])

# x and y coordinates of the point on the surface of the object
# from parameter of the spline
xy_spl = ca.Function('xy_spl',[param_or], [obj_spl.spline(theta_spl(param_or))])

# sqrt(p_x**2 + p_y**2), where p_x  and p_y the x,y coords of the projected point
# The function that computes the Euclidean distance to the point on the surface of the object
dist_spl = ca.Function('dist_spl',[param_or], \
            [ca.sqrt(xy_spl(param_or)[0]**2 + xy_spl(param_or)[1]**2)])

# signed distance of a point to the surface of the object
# dist = dist_p_OnSpline - p_r; - negative when not in contact with the object
dist = ca.Function('dist',[param_or, param_d], [dist_spl(param_or) - param_d ])

#-----------------------------------------------------------------------------#
# This part of the file has basic functions to compute the normal to shape constraint.
# It basically:
#    i) project the angle of a polar point to 0-360 new_range (borrow from above)
#   ii) finds the value of that angle in the range of the spline (borrow from above)
#  iii) computes the dx/dt and dy/dt (tangent), at the angle of the spline
#   iv) normalise tangent
#    v) rotate tangent to global frame (according to objects rot)
#   vi) build normal out of tangent
#
# It utilise the objects module to obtain the spline that describes
# the surface of the object.
# Further, it assumes that the free parameters are polar-coordinates
# of a point w.r.t the centre of the object.
#-----------------------------------------------------------------------------#

# init general params with respect to the normal constraint
param_fiObj = ca.SX.sym("param_fiObj")   # global angle (orientation) of object

# dx/dt and dy/dt coordinates of the point on the surface of the object
# from parameter of the spline
dxdy_spl = ca.Function('xy_spl',[param_or], [obj_spl.spline_jac(theta_spl(param_or))])

# function for the gives the normalised tangent
norm_tang = ca.Function('norm_tang',[param_or], \
            [-1*dxdy_spl(param_or)/ca.sqrt( dxdy_spl(param_or)[0]**2 + dxdy_spl(param_or)[1]**2)])

# tang_global = R*tang_local, where norm_tang are the x,y coords of the tangent vector
# function that rotates the tangent to the global frame
tangent = ca.Function('tangent',[param_or, param_fiObj], \
            [ca.mtimes(oper.R(param_fiObj), norm_tang(param_or))])

# normal_global = [-1*[tangent(2) ; -tangent(1)];]
normal = ca.Function('normal',[param_or, param_fiObj], \
        [-1*ca.vertcat(tangent(param_or, param_fiObj)[1],-1*tangent(param_or, param_fiObj)[0])])


#-----------------------------------------------------------------------------#
# This part of the file has basic functions to compute friction cone manifold
# basis vectors
# ---
# To compute the vertex basis of the friction cone
#       i) we get the global orientation normal at the point of contact.
#      ii) we multiply by -1, cause the normal points outwards and we
#          need it to point inwards
#          so this : -1*normal(param_or, param_fiObj) is the inwards pointing normal w.r.t global frame
#     iii) we compute the rotation matrix to rotate the normal
#          so this : oper.R(param_muFric) is the rotation matrix defined by the friction cone angle
#      iv) we rotate the normal with the rotation matrix to obtain the basis  vectors
#-----------------------------------------------------------------------------#

# init general params with respect to the friction cone
param_muFric = ca.SX.sym("param_muFric")   # angle to normal of friction cone

# Get right edge-vertex(right basis vector) of friction cone
rVertexCone = ca.Function('rVertexCone',[param_or, param_fiObj, param_muFric], \
            [ca.mtimes(oper.R(param_muFric), -1*normal(param_or, param_fiObj))] )

# Get left edge-vertex(left basis vector) of friction cone
lVertexCone = ca.Function('lVertexCone',[param_or, param_fiObj, param_muFric], \
            [ca.mtimes(oper.R(-1*param_muFric), -1*normal(param_or, param_fiObj))] )



# end of file
