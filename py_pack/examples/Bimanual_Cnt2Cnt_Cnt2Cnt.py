#-----------------------------------------------------------------------------#
# This file is an implementation of the Cnt2Cnt - Cnt2Cnt bimanual primitive
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

from py_pack import operators

#-----------------------------------------------------------------------------#

# --------------------------------------------------------------------------
# Specific optimisation problem
# --------------------------------------------------------------------------

# number of control intervals in each phase
intervalsPerPhase = [5, 6]  # implying the knot where the phase is changing

# init parameters
optProb = optVar.defOptVarsConsts(intervalsPerPhase)

# add object vars
optProb.addObjVars()

# add time vars
optProb.addTimeVars()

Aee_name = "HandA"
# add single hand in contact vars
optProb.addContactEndEffVars(Aee_name)

Bee_name = "HandB"
# add single hand in contact vars
optProb.addContactEndEffVars(Bee_name)

# add integration constraints
optProb.addObjIntegConst()

# parameters for the PD controller of the partner
goal = ca.SX.sym("goal",3)    # goal
K = ca.SX.sym("K",3)          # stiffness
D = ca.SX.sym("D",3)          # damping

# add dynamics constraints
optProb.addDynIntegConst2(Aee_name, Bee_name, goal, K, D)

# init variable for contact location of the contact end-effector
Acnt_loc = ca.SX.sym("Acnt_loc")

# we will need to add torque constraint
optProb.addTorqueCntConst(Aee_name, Acnt_loc)

# we will need to add friction cone constraint
optProb.addFrictionConeCntConst(Aee_name, Acnt_loc)

# init variable for contact location of the contact end-effector
Bcnt_loc = ca.SX.sym("Bcnt_loc")

# we will need to add torque constraint
optProb.addTorqueCntConst(Bee_name, Bcnt_loc)

# we will need to add friction cone constraint
optProb.addFrictionConeCntConst(Bee_name, Bcnt_loc)

################# ------------------------------------------------- ############
#  Block to build the cost function
################# ------------------------------------------------- ############

# retrieve variables related with the cost function
pObj, dpObj = optProb.getObjVars()

# cost function weights
cstW = ca.SX.sym("costWeights", 3)

# Build the Cost function
f = cstW[0]*ca.norm_2((goal - pObj[:,optProb.N])**2) + cstW[1]*ca.norm_2(ca.sum1(dpObj[:,:]**2)) + cstW[2]*ca.norm_2(ca.sum1(pObj[:1,:]**2))


################# ------------------------------------------------- ############

# basic nlp solver
nlp = {'x':optProb.varVect, 'p':ca.vertcat(Acnt_loc, Bcnt_loc, goal, K, D, cstW), 'f':f, 'g':optProb.g}

S = ca.nlpsol('S_1Cnt_1Cnt', 'ipopt', nlp, {'ipopt.print_level':5, 'ipopt.hessian_approximation':"exact"})

print("solver is created", S)


# --------------------------------------------------------------------------
# solve the opt problem with initial guess
# --------------------------------------------------------------------------

# retrieve variables related for object
pObj, dpObj = optProb.getObjVars()

# retrieve variables related for time
timeDelta = optProb.getTimeVars()

# retrieve variables related to force of the contact end-effector
LforceCnt, LforceCoefCnt = optProb.getContactEndEffVars(Aee_name)

# retrieve variables related to force of the contact end-effector
RforceCnt, RforceCoefCnt = optProb.getContactEndEffVars(Bee_name)

# build two empty functions for completness of the function library
bPosSw_f = ca.Function('bPosSw_f',[],[])
bVelSw_f = ca.Function('bVelSw_f',[],[])


# obtain functions to build bounds on variables
bPos_f, bVel_f = optProb.buildPosVelBoundFun()
bForceCnt_f = optProb.buildForceBoundFun(Aee_name)
bForceSw_f = optProb.buildForceBoundFun(Bee_name)

# obtain function to build bounds on constraints
bg_f = optProb.buildConstraintBounds()

# build initial guess
x0 = [0.01]*optProb.varVect.size1()

# start at zero (initial state)
init_lbx = 0
# position bounds
max_p = 1

# position bounds
Xlbxf, Xubxf = bPos_f([init_lbx, init_lbx, init_lbx], [-max_p, -max_p, -max_p], [max_p, max_p, max_p])

# initial velocity zero
init_lbdx = 0
# velocity bounds
max_d = 10

dXlbxf, dXubxf = bVel_f([init_lbdx, init_lbdx, init_lbdx], [-max_d, -max_d, -max_d], [max_d, max_d, max_d])

# time bounds
dtlbx = [0.1]*timeDelta.size1()
dtubx = [5]*timeDelta.size1()

# force hand bounds
max_Force = ca.inf
# force coef hand bounds
max_ForceCoeff = 10
# force hand bounds
# force coef hand A bounds
AfCntlbx, AfCntubx, AfcoefCntlbx, AfcoefCntubx = bForceCnt_f(max_Force, max_ForceCoeff)

# force hand bounds
# force coef hand B bounds
BfCntlbx, BfCntubx, BfcoefCntlbx, BfcoefCntubx = bForceCnt_f(max_Force, max_ForceCoeff)


# all variables bounds
lbx =  np.array(Xlbxf)[0,:].tolist() + np.array(dXlbxf)[0,:].tolist() + dtlbx + \
       np.array(AfCntlbx)[0,:].tolist() + np.array(AfcoefCntlbx)[0,:].tolist() + \
       np.array(BfCntlbx)[0,:].tolist() + np.array(BfcoefCntlbx)[0,:].tolist()

ubx =  np.array(Xubxf)[0,:].tolist() + np.array(dXubxf)[0,:].tolist() + dtubx + \
       np.array(AfCntubx)[0,:].tolist() + np.array(AfcoefCntubx)[0,:].tolist() + \
       np.array(BfCntubx)[0,:].tolist() + np.array(BfcoefCntubx)[0,:].tolist()


# bounds for integration constraints should be zero
lbg, ubg = bg_f(0)

# set the goal of the task
goalX = 0.5
goalY = 0.5
goalPhi = ca.pi/2

# parameters of the optimisation problem
# for this problem we have cnt_loc as parameter
p = [ ca.pi, -ca.pi/2, goalX, goalY, goalPhi, 0, 0, 0, 10, 0, 10, 100, 10, 0]


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
print('Index of variables is:', optProb.varVectIndexDict)




#  obtain functions that decode the solution to the physical quantities that we want
getPosF, getVelF, getTimeF, getForcesAF, getForcesBF, getCartAF, getPolarOrientAF, getCartBF, getPolarOrientBF \
                                                = optProb.decodeSolution(Aee_name, Bee_name, SwCnt=False, CntSw=False)


# ----------------------------------------------------------------------------
# extract relevant info from the solution
# ----------------------------------------------------------------------------

pos = x_opt[optProb.varVectIndexDict.get('objPos')[0]:optProb.varVectIndexDict.get('objPos')[-1]]
vel = x_opt[optProb.varVectIndexDict.get('objVel')[0]:optProb.varVectIndexDict.get('objVel')[-1]]
time = np.cumsum( np.append( np.array([0]) , \
       np.array( list( np.array(x_opt[optProb.varVectIndexDict.get('dt')[0]])/optProb.CtrIntValpPhase[0])*optProb.CtrIntValpPhase[0]\
        + list(np.array(x_opt[optProb.varVectIndexDict.get('dt')[-1]-1])/(optProb.CtrIntValpPhase[-1]))*(optProb.CtrIntValpPhase[-1])) ))

Aforces = x_opt[optProb.varVectIndexDict.get("force" + Aee_name)[0]:optProb.varVectIndexDict.get("force" + Aee_name)[-1]]
Bforces = x_opt[optProb.varVectIndexDict.get("force" + Bee_name)[0]:optProb.varVectIndexDict.get("force" + Bee_name)[-1]]

print("Total time of the motion is: ", time[-1])

# the time when the switch of the system dynamics happens
switchKnot_time = np.array(x_opt[optProb.varVectIndexDict.get('dt')[0]])

pos3D = np.array(ca.reshape(pos, 3, int(pos.size1()/3)))
vel3D = np.array(ca.reshape(vel, 3, int(vel.size1()/3)))
Aforces3D = np.array(ca.reshape(Aforces, 3, int(Aforces.size1()/3)))
Bforces3D = np.array(ca.reshape(Bforces, 3, int(Bforces.size1()/3)))


# # # --------------------------------------------------------------------------

#  using decoder functions to obtain solution in physical quantities terms
pos3D = np.array(getPosF(x_opt))
vel3D = np.array(getVelF(x_opt))
t1,t2, timef = getTimeF(x_opt)
time = np.array(timef)
Aforces3D = np.array(getForcesAF(x_opt))
Bforces3D = np.array(getForcesBF(x_opt))

Acart2D = np.array(getCartAF(p[0]))
Bcart2D = np.array(getCartAF(p[1]))


# --------------------------------------------------------------------------
# animate solution trajectory of opt problem
from py_pack import obj_Dist_Norm_Cone

# ------------------------------
# Create the global x and y coordinates of the end-effector
# ------------------------------

_, num_samples =  pos3D.shape

# compute local transform
Lee_pos = []
Ree_pos = []
for i in range(num_samples):
    Lee_pos.append(np.array(ca.mtimes(operators.R(pos3D[2,i]), Acart2D[:,i].T)))

    Ree_pos.append(np.array(ca.mtimes(operators.R(pos3D[2,i]), Bcart2D[:,i].T)))

# left ee
Aee_posNp = np.array(Lee_pos)
d1,d2,_ = Aee_posNp.shape
Aee_posNp = Aee_posNp.reshape(d1, d2)
# add local transform to the location of the object
Aee_posNpGlobal = Aee_posNp.T + pos3D[0:2,:]

# right ee
Bee_posNp = np.array(Ree_pos)
d1,d2,_ = Bee_posNp.shape
Bee_posNp = Bee_posNp.reshape(d1, d2)
# add local transform to the location of the object
Bee_posNpGlobal = Bee_posNp.T + pos3D[0:2,:]

# animate
animTraj.animationfunc2Hands(pos3D[0,:], pos3D[1,:], pos3D[2,:], \
                       Aee_posNpGlobal[0,:], Aee_posNpGlobal[1,:],\
                       Aforces3D[0,:], Aforces3D[1,:], \
                       Bee_posNpGlobal[0,:], Bee_posNpGlobal[1,:],\
                       Bforces3D[0,:], Bforces3D[1,:],\
                       time[-1]/optProb.N,\
                       goalX, goalY, goalPhi)




# end of file
