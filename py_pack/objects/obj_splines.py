####################
# Extrenal libs
####################
# casadi
from py_pack import ca
# numpy
from py_pack import np
# yaml
import yaml

# ====================================
#    Spline Basis function
# source : https://github.com/casadi/casadi/issues/1484
#          https://gist.github.com/jgillis/54767ae9e38dca3dfcb9144fb4eb4398
# =====================================
with open("py_pack/config/parameters.yml", 'r') as ymlfile:
    params = yaml.load(ymlfile, Loader=yaml.FullLoader)

t = ca.SX.sym("t")

if params['object']['shape']['shapeID'] == 0:


    # Coeficients of the spline in pp form (MATLAB)
    coefMat = np.array([ [0.0915,   -0.4237,    0.0000,    0.2650],\
                         [-0.2221,   -0.0000,    0.4612,         0],\
                         [0.2221,   -0.3001,   -0.3260,    0.1874],\
                         [-0.0915,   -0.3001,    0.3260,    0.1874],\
                         [0.2221,   -0.0000,   -0.4612,         0],\
                         [0.0915,   -0.4237,    0.0000,    0.2650],\
                         [0.0915,    0.3001,   -0.3260,   -0.1874],\
                         [0.2221,   -0.3001,   -0.3260,    0.1874],\
                         [-0.0915,    0.4237,         0,   -0.2650],\
                         [0.2221,    0.0000,   -0.4612,         0],\
                         [-0.2221,    0.3001,    0.3260,   -0.1874],\
                         [0.0915,    0.3001,   -0.3260,   -0.1874],\
                         [-0.2221,   -0.0000,    0.4612,         0],\
                         [-0.0915,    0.4237,         0,   -0.2650],\
                         [-0.0915,   -0.3001,    0.3260,    0.1874],\
                         [-0.2221,    0.3001,    0.3260,   -0.1874]])

    # circle with radius 0.265
    spl_brk = [0, 0.4504, 0.9007, 1.3511, 1.8015, 2.2519, 2.7022, 3.1526, 3.6030]


elif params['object']['shape']['shapeID'] == 1:


    # Coeficients of the spline in pp form (MATLAB)
    #######################################################
    coefMat = np.array([[-0.8571,    0.4286,         0,    0.2500],\
                        [-0.5714,   -0.0000,    0.6429,         0],\
                        [ 0.5714,   -0.8571,   -0.2143,    0.2500],\
                        [ 0.8571,   -0.8571,    0.2143,    0.2500],\
                        [ 0.5714,    0.0000,   -0.6429,        0],\
                        [-0.8571,    0.4286,    0.0000,    0.2500],\
                        [-0.8571,    0.8571,   -0.2143,   -0.2500],\
                        [ 0.5714,   -0.8571,   -0.2143,    0.2500],\
                        [ 0.8571,   -0.4286,         0,   -0.2500],\
                        [ 0.5714,    0.0000,   -0.6429,         0],\
                        [-0.5714,    0.8571,    0.2143,   -0.2500],\
                        [-0.8571,    0.8571,   -0.2143,   -0.2500],\
                        [-0.5714,   -0.0000,    0.6429,         0],\
                        [ 0.8571,   -0.4286,         0,   -0.2500],\
                        [ 0.8571,   -0.8571,    0.2143,    0.2500],\
                        [-0.5714,    0.8571,    0.2143,   -0.2500]])

    spl_brk = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


elif params['object']['shape']['shapeID'] == 2:

    coefMat = np.array([[-1.2252,    0.4901,   -0.0000,    0.3200],\
                        [-0.6066,   -0.0000,    0.4971,         0],\
                        [ 0.5776,   -0.9802,  -0.1960,    0.3200],\
                        [ 0.6434,   -0.7279,    0.2059,    0.1600],\
                        [ 0.5776,   -0.0000,   -0.7505,         0],\
                        [-0.6434,    0.3640,    0.0000,    0.1600],\
                        [-1.2252,    0.9802,   -0.1960,   -0.3200],\
                        [ 0.6066,   -0.7279,   -0.2059,    0.1600],\
                        [ 1.2252,   -0.4901,    0.0000,   -0.3200],\
                        [ 0.6066,    0.0000,   -0.4971,         0],\
                        [-0.5776,    0.9802,    0.1960,   -0.3200],\
                        [-0.6434,    0.7279,   -0.2059,   -0.1600],\
                        [-0.5776,   -0.0000,    0.7505,         0],\
                        [ 0.6434,   -0.3640,         0,   -0.1600],\
                        [ 1.2252,   -0.9802,    0.1960,    0.3200],\
                        [-0.6066,   0.7279,     0.2059,   -0.1600]])


    spl_brk = [0, 0.4000, 0.9657, 1.5314, 1.9314, 2.3314, 2.8971, 3.4627, 3.8627]


# X coeffs obtained from MATLAB are the odd rows
x_coefMat = coefMat[0:-1:2,:]
rel_x = ca.Function('rel_x',[t],[ca.if_else(t<=spl_brk[1], x_coefMat[0,0]*t**3 + x_coefMat[0,1]*t**2 + x_coefMat[0,2]*t +x_coefMat[0,3],\
                                ca.if_else(t<=spl_brk[2], x_coefMat[1,0]*(t-spl_brk[1])**3 + x_coefMat[1,1]*(t-spl_brk[1])**2 + x_coefMat[1,2]*(t-spl_brk[1]) +x_coefMat[1,3],\
                                ca.if_else(t<=spl_brk[3], x_coefMat[2,0]*(t-spl_brk[2])**3 + x_coefMat[2,1]*(t-spl_brk[2])**2 + x_coefMat[2,2]*(t-spl_brk[2]) +x_coefMat[2,3],\
                                ca.if_else(t<=spl_brk[4], x_coefMat[3,0]*(t-spl_brk[3])**3 + x_coefMat[3,1]*(t-spl_brk[3])**2 + x_coefMat[3,2]*(t-spl_brk[3]) +x_coefMat[3,3],\
                                ca.if_else(t<=spl_brk[5], x_coefMat[4,0]*(t-spl_brk[4])**3 + x_coefMat[4,1]*(t-spl_brk[4])**2 + x_coefMat[4,2]*(t-spl_brk[4]) +x_coefMat[4,3],\
                                ca.if_else(t<=spl_brk[6], x_coefMat[5,0]*(t-spl_brk[5])**3 + x_coefMat[5,1]*(t-spl_brk[5])**2 + x_coefMat[5,2]*(t-spl_brk[5]) +x_coefMat[5,3],\
                                ca.if_else(t<=spl_brk[7], x_coefMat[6,0]*(t-spl_brk[6])**3 + x_coefMat[6,1]*(t-spl_brk[6])**2 + x_coefMat[6,2]*(t-spl_brk[6]) +x_coefMat[6,3],\
                                x_coefMat[7,0]*(t-spl_brk[7])**3 + x_coefMat[7,1]*(t-spl_brk[7])**2 + x_coefMat[7,2]*(t-spl_brk[7]) +x_coefMat[7,3],)))))))])

# Y coeffs obtained from MATLAB are the even rows
y_coefMat = coefMat[1::2,:]
rel_y = ca.Function('rel_y',[t],[ca.if_else(t<=spl_brk[1], y_coefMat[0,0]*t**3 + y_coefMat[0,1]*t**2 + y_coefMat[0,2]*t +y_coefMat[0,3],\
                                ca.if_else(t<=spl_brk[2], y_coefMat[1,0]*(t-spl_brk[1])**3 + y_coefMat[1,1]*(t-spl_brk[1])**2 + y_coefMat[1,2]*(t-spl_brk[1]) +y_coefMat[1,3],\
                                ca.if_else(t<=spl_brk[3], y_coefMat[2,0]*(t-spl_brk[2])**3 + y_coefMat[2,1]*(t-spl_brk[2])**2 + y_coefMat[2,2]*(t-spl_brk[2]) +y_coefMat[2,3],\
                                ca.if_else(t<=spl_brk[4], y_coefMat[3,0]*(t-spl_brk[3])**3 + y_coefMat[3,1]*(t-spl_brk[3])**2 + y_coefMat[3,2]*(t-spl_brk[3]) +y_coefMat[3,3],\
                                ca.if_else(t<=spl_brk[5], y_coefMat[4,0]*(t-spl_brk[4])**3 + y_coefMat[4,1]*(t-spl_brk[4])**2 + y_coefMat[4,2]*(t-spl_brk[4]) +y_coefMat[4,3],\
                                ca.if_else(t<=spl_brk[6], y_coefMat[5,0]*(t-spl_brk[5])**3 + y_coefMat[5,1]*(t-spl_brk[5])**2 + y_coefMat[5,2]*(t-spl_brk[5]) +y_coefMat[5,3],\
                                ca.if_else(t<=spl_brk[7], y_coefMat[6,0]*(t-spl_brk[6])**3 + y_coefMat[6,1]*(t-spl_brk[6])**2 + y_coefMat[6,2]*(t-spl_brk[6]) +y_coefMat[6,3],\
                                y_coefMat[7,0]*(t-spl_brk[7])**3 + y_coefMat[7,1]*(t-spl_brk[7])**2 + y_coefMat[7,2]*(t-spl_brk[7]) +y_coefMat[7,3],)))))))])



# cubic-order splines in pp form (MATLAB)
# re-parametrise the spline
u = ca.SX.sym("u")
# concatanate the two dimensions
lhs = ca.horzcat(rel_x(u), rel_y(u))
# build the spline function
spline = ca.Function('spline',[u], [lhs])

# build the derivative of the spline function
spl_jac = ca.jacobian(spline(u), u)
spline_jac = ca.Function('spline_jac',[u], [spl_jac])
