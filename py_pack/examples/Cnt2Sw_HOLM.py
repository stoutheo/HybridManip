#-----------------------------------------------------------------------------#
# This file is an implementation of the Cnt2Sw primitive
#   ......
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
# build optimisation problem
from py_pack import optVar

# build animate solution trajectory
from py_pack import animTraj

# functions related to the shape of the object
from py_pack import obj_Dist_Norm_Cone

from py_pack import operators

#-----------------------------------------------------------------------------#

# --------------------------------------------------------------------------
# Specific optimisation problem
# --------------------------------------------------------------------------

# number of control intervals in each phase
intervalsPerPhase = [6, 6]  # implying the knot where the phase is changing

# init parameters
optProb = optVar.defOptVarsConsts(intervalsPerPhase)

# add object vars
optProb.addObjVars()

# add time vars
optProb.addTimeVars()

ee_name = "leftHand"
# add single hand in contact vars
optProb.addContactSwingEndEffVars(ee_name)

# add integration constraints
optProb.addObjIntegConst()

# parameters for the PD controller of the partner
goal = ca.SX.sym("goal",3)    # goal
K = ca.SX.sym("K",3)          # stiffness
D = ca.SX.sym("D",3)          # damping

# add dynamics constraints
optProb.addDynIntegConst(ee_name, goal, K, D)


# init variable for contact location of the contact end-effector
cnt_loc = ca.SX.sym("cnt_loc")

# add torque constraint
optProb.addTorqueCntConst(ee_name, cnt_loc)

# add friction cone constraint
optProb.addFrictionConeCntConst(ee_name, cnt_loc)

# add distance from the shape constraint
optProb.addSwingContConst(ee_name)

# add integration constraint for the contact-swing end effector
optProb.addSwingLocalIntegConst(ee_name, cnt_loc)


################# ------------------------------------------------- ############
#  Block to build the cost function
################# ------------------------------------------------- ############

# retrieve variables related with the cost function
pObj, _ = optProb.getObjVars()

# Build the Cost function
f = ca.norm_2((goal - pObj[:,optProb.N])**2)
################# ------------------------------------------------- ############

# basic nlp solver
nlp = {'x':optProb.varVect, 'p':ca.vertcat(cnt_loc, goal, K, D), 'f':f, 'g':optProb.g}

S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt.print_level':5, 'ipopt.hessian_approximation':"exact"})

print("solver is created", S)

# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
# solve the opt problem with initial guess
# #

# retrieve variables related for object
pObj, dpObj = optProb.getObjVars()

# retrieve variables related for time
timeDelta = optProb.getTimeVars()

# retrieve variables related to force of the contact end-effector
forceCnt, forceCoefCnt, polarPosSw, polarVelSw = optProb.getContactSwingEndEff(ee_name)

# build initial guess
x0 = [0]*optProb.varVect.size1()

# start at zero (initial state)
init_lbx = 0

# an other option is to put the final condition in the lbx and ubx

# position bounds
Xlbx = [-init_lbx, -init_lbx, -init_lbx] + [-ca.inf]*(pObj.size2()-1)*pObj.size1()# + [goalX, goalY, goalPhi]
Xubx = [init_lbx, init_lbx, init_lbx] + [ca.inf]*(pObj.size2()-1)*pObj.size1()# + [goalX, goalY, goalPhi]

# velocity bounds
max_d = 10
# initial velocity zero
dXlbx = [0,0,0] + [-max_d]*(dpObj.size2()-1)*dpObj.size1()
dXubx = [0,0,0] + [max_d]*(dpObj.size2()-1)*dpObj.size1()

# time bounds
dtlbx = [0.1]*timeDelta.size1()
dtubx = [1.5]*timeDelta.size1()

# force hand bounds
fCntlbx = [-ca.inf]*forceCnt.size1()*forceCnt.size2()
fCntubx = [ca.inf]*forceCnt.size1()*forceCnt.size2()

# force coef hand bounds
fcoefCntlbx = [0]*forceCoefCnt.size1()*forceCoefCnt.size2()
fcoefCntubx = [10]*forceCoefCnt.size1()*forceCoefCnt.size2()

# relative coordinate bounds of the position of the swing hand
polarPoslbx = [-ca.inf]*polarPosSw.size1()*polarPosSw.size2()
polarPosubx = [ca.inf]*polarPosSw.size1()*polarPosSw.size2()

# relative coordinate bounds of the velocity of the swing hand
polarVellbx = [-max_d]*polarVelSw.size1()*polarVelSw.size2()
polarVelubx = [max_d]*polarVelSw.size1()*polarVelSw.size2()

# all variables bounds
lbx =  Xlbx + dXlbx + dtlbx + fCntlbx + fcoefCntlbx + polarPoslbx + polarVellbx
ubx =  Xubx + dXubx + dtubx + fCntubx + fcoefCntubx + polarPosubx + polarVelubx

# get the bounds of the constraints
_, gBounds = optProb.getConstraints()

# bounds for integration constraints should be zero
lbg = gBounds[0,:]
ubg = gBounds[1,:]

# set the goal of the task
goalX = 0.8
goalY = -0.2
goalPhi = ca.pi/2
# parameters of the optimisation problem
# for this problem we have cnt_loc as parameter
p = [3*ca.pi/2, goalX, goalY, goalPhi, 10, 10, 0, 10, 10, 10] #[ca.pi + ca.pi/4]


r = S(x0=x0,\
      p=p, \
      lbx = lbx, \
      ubx = ubx, \
      lbg = lbg, \
      ubg = ubg)

print("Number of variables are : ", optProb.varVect.size1())

# # obtaining the stats of the solver
statsDict = S.stats()
print("Did the solver succeed?: ", statsDict.get('success'))

x_opt = r['x']
print("1st phase time =", x_opt[optProb.varVectIndexDict.get('dt')[0]] ,  "and for 2nd phase time = ", x_opt[optProb.varVectIndexDict.get('dt')[-1]-1])


# ----------------------------------------------------------------------------
# extract relevant info from the solution

#################

pos = x_opt[optProb.varVectIndexDict.get('objPos')[0]:optProb.varVectIndexDict.get('objPos')[-1]]
vel = x_opt[optProb.varVectIndexDict.get('objVel')[0]:optProb.varVectIndexDict.get('objVel')[-1]]
time = np.cumsum( np.append( np.array([0]) , \
       np.array( list( np.array(x_opt[optProb.varVectIndexDict.get('dt')[0]])/optProb.CtrIntValpPhase[0])*optProb.CtrIntValpPhase[0]\
        + list(np.array(x_opt[optProb.varVectIndexDict.get('dt')[-1]-1])/(optProb.CtrIntValpPhase[-1]))*(optProb.CtrIntValpPhase[-1])) ))

forces = x_opt[optProb.varVectIndexDict.get("force" + ee_name)[0]:optProb.varVectIndexDict.get("force" + ee_name)[-1]]

polar = x_opt[optProb.varVectIndexDict.get("polarPos" + ee_name)[0]:optProb.varVectIndexDict.get("polarPos" + ee_name)[-1]]

print("Total time of the motion is: ", time[-1])
# the time when the switch of the system dynamics happens
switchKnot_time = np.array(x_opt[optProb.varVectIndexDict.get('dt')[0]])

pos3D = np.array(ca.reshape(pos, 3, int(pos.size1()/3)))
vel3D = np.array(ca.reshape(vel, 3, int(vel.size1()/3)))
forces3D = np.array(ca.reshape(forces, 3, int(forces.size1()/3)))
# append force vector with zeros for the swing phase of the end effector
forces3D = np.hstack((forces3D,np.tile(np.zeros(3),[3, int(pos.size1()/3) - int(forces.size1()/3)])))

#obtain polar coord
polar2D = np.array(ca.reshape(polar, 2, int(polar.size1()/2)))

# --------------------------------------------------------------------------

# ------------------------------
# Create the global x and y coordinates of the end-effector
# ------------------------------

num_samples =  optProb.N + 1

# compute local transform
ee_pos = []
for i in range(num_samples):

    # if in contact with the object
    if i <= optProb.CtrIntValpPhase[0]:
        ee_pos.append(np.array(ca.mtimes(operators.R(pos3D[2,i]), obj_Dist_Norm_Cone.xy_spl(p[0]).T)))
    else:   # if in swing motion
        idx = i - optProb.CtrIntValpPhase[0] - 1
        ee_pos.append(np.array(ca.mtimes(operators.R(pos3D[2,i]), operators.circlePt(polar2D[1,idx], polar2D[0,idx]))))


ee_posNp = np.array(ee_pos)
d1,d2,_ = ee_posNp.shape
ee_posNp = ee_posNp.reshape(d1, d2)

# add local transform to the location of the object
ee_posNpGlobal = ee_posNp.T + pos3D[0:2,:]



# animate
animTraj.animationfunc1Hand(pos3D[0,:], pos3D[1,:], pos3D[2,:], \
                       ee_posNpGlobal[0,:], ee_posNpGlobal[1,:],\
                       forces3D[0,:], forces3D[1,:], \
                       time[-1]/optProb.N,\
                       goalX, goalY, goalPhi)

# end of file
