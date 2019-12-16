#-----------------------------------------------------------------------------#
# This file is the back-end for building the HOLM primitives
# it defines all the variables of the optimisation problem
# ......
#-----------------------------------------------------------------------------#


####################
# Extrenal libs
####################
# casadi
from py_pack import ca
# numpy
from py_pack import np
# yaml
import yaml

####################
# Custom package libs
####################
# integration constraint
from py_pack import con_Intgr

# torque constraint
from py_pack import con_Tau

# friction cone constraint
from py_pack import con_Vfc

# distance to object
from py_pack import obj_Dist_Norm_Cone

# elementary functions
from py_pack import operators

#-----------------------------------------------------------------------------#

class defOptVarsConsts:
    """
    Description of the class...
    """


    def __init__(self, phaseCtrIntval, objIndex=1):
        """
         Init parameters
        """

        with open("py_pack/config/parameters.yml", 'r') as ymlfile:
            params = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # number of control intervals per phase
        self.N = sum(phaseCtrIntval)

        # number of phases of the trajectory
        self.numPh = len(phaseCtrIntval)

        # number of control intervals per phases of the trajectory
        self.CtrIntValpPhase = phaseCtrIntval

        # list and dict of the variables in the symbolic form
        self.varList = []
        self.varIndexDict = {}

        # initialise the index of vector form of the variables
        self.lastIndex = 0

        # list and dict for the vector form of the variables
        self.varVectIndexDict = {}
        self.varVect = ca.vertcat([])

        # constraint mat
        self.g_boundList = [[],[]]
        self.g = ca.vertcat([])

        # gravity vector
        self.grav = np.array(params['gravity'])

        # object physical properties
        if params['object']['shape']['shapeID'] == 0:
            self.radius = params['object']['shape']['circle']['radius']
            self.Inertia = params['object']['shape']['circle']['inertia']

        if params['object']['shape']['shapeID'] == 1:
            self.Inertia = params['object']['shape']['rectangle']['inertia']

        if params['object']['shape']['shapeID'] == 2:
            self.Inertia = params['object']['shape']['parallelogram']['inertia']

        # set the mass
        self.mass  = params['object']['mass']

        # set the friction coefficient
        mu = params['object']['friction']
        self.muCone = ca.atan(mu)

        # set the maximum allowed distance of a end effector during swing
        self.maxSwDist = params['constraints']['maxDistatSwing']

        # use parallel threads (no gain appears)
        self.threads = 1

    ### -------------------------------------------------------------------------###
    ###                       Functions for variable declaration                 ###
    ### -------------------------------------------------------------------------###

    def addObjVars(self):
        """
         Add object related variables
        """

        # init variables ------for the object---------
        # position of the object in 2D
        objPosName = "objPos"
        objPos = ca.SX.sym(objPosName, 3, self.N + 1)
        end_objPos = self.lastIndex + 3*(self.N + 1)

        # velocity of the object in 2D
        dobjPosName = "objVel"
        dobjPos = ca.SX.sym(dobjPosName, 3, self.N + 1)
        end_dobjPos = end_objPos + 3*(self.N + 1)

        # append all the variables in a list
        self.varIndexDict[objPosName] = len(self.varList)
        self.varList.append(objPos)
        self.varIndexDict[dobjPosName] = len(self.varList)
        self.varList.append(dobjPos)


        # reshape SX mats to vectors to give them to the optimizer
        objPosVector = ca.reshape(objPos, objPos.size1()*objPos.size2(), 1)
        dobjPosVector = ca.reshape(dobjPos, dobjPos.size1()*dobjPos.size2(), 1)

        # group all the variables in a vector
        self.varVect = ca.vertcat(self.varVect, \
                                  objPosVector, dobjPosVector)

        # generate the dict with names and indexs of these qualities in the vector
        self.varVectIndexDict = { objPosName : [self.lastIndex, end_objPos],
                                  dobjPosName : [end_objPos,end_dobjPos]}

        # keep track of the latest index of quantities
        self.lastIndex = end_dobjPos

        # declaring class accessible variables to be used from other function
        self.objPosName = objPosName
        self.dobjPosName = dobjPosName


    def addTimeVars(self):
        """
         Add time variables which depend on the number of phases
        """

        # dt of each phases of the trajectory
        dtname = "dt"
        dt = ca.SX.sym(dtname, self.numPh)
        edt = self.lastIndex + self.numPh

        # append all the variables in a list
        self.varIndexDict[dtname] = len(self.varList)
        self.varList.append(dt)

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, dt)
        self.varVectIndexDict[dtname] = [self.lastIndex, edt]

        # keep track of the latest index of quantities
        self.lastIndex = edt

        # declaring class accessible variables to be used from other function
        self.dtname = dtname


    def addContactEndEffVars(self, name):
        """
         Add end-effector being only in contact
        """

        # init variables ------for contact end effector ---------

        # force related params
        forceCntname = "force" + name
        forceCont = ca.SX.sym(forceCntname, 3, self.N + 1)
        eforce = self.lastIndex + 3*(self.N + 1)

        # coeefficients a and b
        coefsCntname = "f_coefs" + name
        f_coefsCont = ca.SX.sym(coefsCntname, 2, self.N + 1)
        ef_coefs = eforce + 2*(self.N + 1)

        # append all the variables in a list
        self.varIndexDict[forceCntname] = len(self.varList)
        self.varList.append(forceCont)
        self.varIndexDict[coefsCntname] = len(self.varList)
        self.varList.append(f_coefsCont)

        # save the knots in which the forces are applied on the object
        self.varIndexDict["forceKnotIdx" + name] = [0, self.N]

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, \
                                  ca.reshape(forceCont, forceCont.size1()*forceCont.size2(), 1),\
                                  ca.reshape(f_coefsCont, f_coefsCont.size1()*f_coefsCont.size2(), 1))
        self.varVectIndexDict[forceCntname] = [self.lastIndex, eforce]
        self.varVectIndexDict[coefsCntname] = [eforce, ef_coefs]

        # keep track of the latest index of quantities
        self.lastIndex = ef_coefs



    def addContactSwingEndEffVars(self, name):
        """
         Add end-effector being 1st in contact and then swinging
        """

        # ------------------ Contact phase ------------------------------------#

        # get the number of control intervals for the contact phase
        cntIntvals = self.CtrIntValpPhase[0]

        # force related params
        forceCntSwname = "force" + name
        forceContSw = ca.SX.sym(forceCntSwname, 3, cntIntvals + 1)
        eforce = self.lastIndex + 3*(cntIntvals + 1)

        # coeefficients a and b
        coefsCntSwname = "f_coefs" + name
        f_coefsContSw = ca.SX.sym(coefsCntSwname, 2, cntIntvals + 1)
        ef_coefs = eforce + 2*(cntIntvals + 1)

        # append all the variables in a list
        self.varIndexDict[forceCntSwname] = len(self.varList)
        self.varList.append(forceContSw)
        self.varIndexDict[coefsCntSwname] = len(self.varList)
        self.varList.append(f_coefsContSw)

        # save the knots in which knors the forces are applied on the object
        self.varIndexDict["forceKnotIdx" + name] = [0, cntIntvals]

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, \
                                  ca.reshape(forceContSw, forceContSw.size1()*forceContSw.size2(), 1),\
                                  ca.reshape(f_coefsContSw, f_coefsContSw.size1()*f_coefsContSw.size2(), 1))
        self.varVectIndexDict[forceCntSwname] = [self.lastIndex, eforce]
        self.varVectIndexDict[coefsCntSwname] = [eforce, ef_coefs]

        # keep track of the latest index of quantities
        self.lastIndex = ef_coefs

        # ------------------ Swing phase --------------------------------------#

        try:
            # get the number of control intervals for the swing phase
            cntIntvals = self.CtrIntValpPhase[1]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' +  \
            ': Optimization problem does not have a second phase \n')
            exit()

        # polar position related params
        polarCntSwname = "polarPos" + name
        # polar-coord angle  and polar-coord distance
        polarContSwing = ca.SX.sym(polarCntSwname, 2, cntIntvals)
        epolar = self.lastIndex + 2*(cntIntvals)

        # polar velocity related params
        dpolarCntSwname = "polarVel" + name
        # polar-coord angle  and polar-coord distance
        dpolarContSwing = ca.SX.sym(dpolarCntSwname, 2, cntIntvals)
        edpolar = epolar + 2*(cntIntvals)

        # append all the variables in a list
        self.varIndexDict[polarCntSwname] = len(self.varList)
        self.varList.append(polarContSwing)
        self.varIndexDict[dpolarCntSwname] = len(self.varList)
        self.varList.append(dpolarContSwing)

        # save the knots in which knors the forces are applied on the object
        self.varIndexDict["polarKnotIdx" + name] = [self.CtrIntValpPhase[0], self.CtrIntValpPhase[0] + cntIntvals]

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, \
                                  ca.reshape(polarContSwing, polarContSwing.size1()*polarContSwing.size2(), 1),\
                                  ca.reshape(dpolarContSwing, dpolarContSwing.size1()*dpolarContSwing.size2(), 1))
        self.varVectIndexDict[polarCntSwname] = [self.lastIndex, epolar]
        self.varVectIndexDict[dpolarCntSwname] = [epolar, edpolar]

        # keep track of the latest index of quantities
        self.lastIndex = edpolar


    def addSwingContactEndEffVars(self, name):
        """
         Add end-effector being 1st swinging and then in contact
        """

        # ------------------ Swing phase --------------------------------------#

        try:
            # get the number of control intervals for the swing phase
            cntIntvals = self.CtrIntValpPhase[0]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' +  \
            ': Optimization problem does not have a first phase \n')
            exit()

        # polar position related params
        polarCntSwname = "polarPos" + name
        # polar-coord angle  and polar-coord distance
        polarContSwing = ca.SX.sym(polarCntSwname, 2, cntIntvals)
        epolar = self.lastIndex + 2*(cntIntvals)

        # polar velocity related params
        dpolarCntSwname = "polarVel" + name
        # polar-coord angle  and polar-coord distance
        dpolarContSwing = ca.SX.sym(dpolarCntSwname, 2, cntIntvals)
        edpolar = epolar + 2*(cntIntvals)

        # append all the variables in a list
        self.varIndexDict[polarCntSwname] = len(self.varList)
        self.varList.append(polarContSwing)
        self.varIndexDict[dpolarCntSwname] = len(self.varList)
        self.varList.append(dpolarContSwing)

        # save the knots in which knors the forces are applied on the object
        self.varIndexDict["polarKnotIdx" + name] = [0, cntIntvals]

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, \
                                  ca.reshape(polarContSwing, polarContSwing.size1()*polarContSwing.size2(), 1),\
                                  ca.reshape(dpolarContSwing, dpolarContSwing.size1()*dpolarContSwing.size2(), 1))
        self.varVectIndexDict[polarCntSwname] = [self.lastIndex, epolar]
        self.varVectIndexDict[dpolarCntSwname] = [epolar, edpolar]

        # keep track of the latest index of quantities
        self.lastIndex = edpolar


        # ------------------ Contact phase ------------------------------------#

        # get the number of control intervals for the contact phase
        cntIntvals = self.CtrIntValpPhase[1]

        # force related params
        forceCntSwname = "force" + name
        forceContSw = ca.SX.sym(forceCntSwname, 3, cntIntvals + 1)
        eforce = self.lastIndex + 3*(cntIntvals + 1)

        # coeefficients a and b
        coefsCntSwname = "f_coefs" + name
        f_coefsContSw = ca.SX.sym(coefsCntSwname, 2, cntIntvals + 1)
        ef_coefs = eforce + 2*(cntIntvals + 1)

        # append all the variables in a list
        self.varIndexDict[forceCntSwname] = len(self.varList)
        self.varList.append(forceContSw)
        self.varIndexDict[coefsCntSwname] = len(self.varList)
        self.varList.append(f_coefsContSw)

        # save the knots in which knors the forces are applied on the object
        self.varIndexDict["forceKnotIdx" + name] = [self.CtrIntValpPhase[0], self.CtrIntValpPhase[0] + cntIntvals]

        # update the list and dict of vars with new variables
        self.varVect = ca.vertcat(self.varVect, \
                                  ca.reshape(forceContSw, forceContSw.size1()*forceContSw.size2(), 1),\
                                  ca.reshape(f_coefsContSw, f_coefsContSw.size1()*f_coefsContSw.size2(), 1))
        self.varVectIndexDict[forceCntSwname] = [self.lastIndex, eforce]
        self.varVectIndexDict[coefsCntSwname] = [eforce, ef_coefs]

        # keep track of the latest index of quantities
        self.lastIndex = ef_coefs

    ### -------------------------------------------------------------------------###
    ###                  Functions for accessing the opt variables               ###
    ### -------------------------------------------------------------------------###
    def getVars(self):
        """
        Returns all optimisation variables
        """
        return self.varList, self.varIndexDict


    def getObjVars(self):
        """
        Returns the object related variables
        """
        try:
            objPosInd = self.varIndexDict.get(self.objPosName)
            objVelInd = self.varIndexDict.get(self.dobjPosName)
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' +  \
            ': Optimization problem does not have object variables initialised \n')
            exit()

        return self.varList[objPosInd], self.varList[objVelInd]


    def getTimeVars(self):
        """
        Returns the time related variables
        """
        try:
            timeInd = self.varIndexDict.get(self.dtname)
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have time variables initialised \n')
            exit()

        return self.varList[timeInd]


    def getContactEndEffVars(self, name):
        """
        Returns the force variables of contact end effector
        """
        try:
            forceCnt = self.varIndexDict.get("force" + name)
            handCntForceCoeffs = self.varIndexDict.get("f_coefs" + name)
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have contact hand force variables or force coefficients initialised \n')
            exit()

        return self.varList[forceCnt], self.varList[handCntForceCoeffs]


    def getContactSwingEndEff(self, name):
        """
        Returns all the variables of contact-swing end effector
        """
        try:
            forceCnt = self.varIndexDict.get("force" + name)
            handCntForceCoeffs = self.varIndexDict.get("f_coefs" + name)
            handSwPolar = self.varIndexDict.get("polarPos" + name)
            dhandSwPolar = self.varIndexDict.get("polarVel" + name)
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have contact hand force variables or force coefficients initialised \n')
            exit()

        return self.varList[forceCnt], self.varList[handCntForceCoeffs], self.varList[handSwPolar], self.varList[dhandSwPolar]


    def getConstraints(self):
        """
        Returns the the constraints generated along with their bounds
        """
        return self.g, np.array(self.g_boundList)


    ### -------------------------------------------------------------------------###
    ###                       Functions for constraint building                  ###
    ### -------------------------------------------------------------------------###

    def addObjIntegConst(self):
        """
         Add integration constraints of the object motion
        """
        try:
            # get object related symbolic variables
            objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            objVel = self.varList[self.varIndexDict.get(self.dobjPosName)]
            dtime = self.varList[self.varIndexDict.get(self.dtname)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or time variables initialised \n')
            exit()

        # integrate position from velocities for each phase
        prev_knotsPhase = 0
        for numPhase in range(self.numPh):
            numknotsPhase = self.CtrIntValpPhase[numPhase]      # number of knots in this phase
            knotsPhase = prev_knotsPhase + numknotsPhase        # index to end knot of the phase

            # aggregate integration function as many times as the number of knots in this phase
            TrapAll3D = con_Intgr.trap_quad3D.map(numknotsPhase, "thread", self.threads)

            # build the constraints
            gInter = TrapAll3D(objPos[:,prev_knotsPhase:knotsPhase], objPos[:,prev_knotsPhase+1:knotsPhase+1], \
                               objVel[:,prev_knotsPhase:knotsPhase], objVel[:,prev_knotsPhase+1:knotsPhase+1], \
                               dtime[numPhase]/float(numknotsPhase))

            # index to end knot of the phase, which will be the first knot of the next phase
            prev_knotsPhase = knotsPhase

            # gInter.size1()*gInter.size2() are the number of bounds as many as the constraints
            # and zero [0] as the bound of the integration constraints has to be zero
            # lower bound
            self.g_boundList[0] += [0]*gInter.size1()*gInter.size2()
            # upper bound
            self.g_boundList[1] += [0]*gInter.size1()*gInter.size2()

            # build the constraint mat
            self.g = ca.vertcat(self.g, ca.reshape(gInter,gInter.size1()*gInter.size2(),1))


    def addDynIntegConst(self, name, goal, K, D):
        """
         Add dynamics constraints of the object motion, w.r.t to forces applied on the object
         by one end effector
        """

        try:
            # get object related symbolic variables
            objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            objVel = self.varList[self.varIndexDict.get(self.dobjPosName)]
            dtime = self.varList[self.varIndexDict.get(self.dtname)]
            handCntForce = self.varList[self.varIndexDict.get("force" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or time or contact hand force variables initialised \n')
            exit()

        # obtain the index in terms of knots in which these forces apply on
        handCntForceIdx = self.varIndexDict.get("forceKnotIdx" + name)

        # integrate velocities from forces for each phase
        prev_knotsPhase = 0
        for numPhase in range(self.numPh):
            numknotsPhase = self.CtrIntValpPhase[numPhase]      # number of knots in this phase
            knotsPhase = prev_knotsPhase + numknotsPhase        # index to end knot of the phase

            # aggregate dynamics function as many times as the number of knots in this phase
            DynamTq3D = con_Intgr.dynam_tq3D.map(numknotsPhase, "thread", self.threads)

            # create the goal matrices for the all the knots of the phase
            goalMat = ca.SX(np.tile(np.array([goal[0],goal[1],goal[2]]),[numknotsPhase, 1]).T)

            # compute the PD force at the 1st point of the trapezoidal for all the knots of the phase
            posErrMat_cur = ( goalMat - objPos[:,prev_knotsPhase:knotsPhase])
            PD_cur =  K*posErrMat_cur + D*(-objVel[:,prev_knotsPhase:knotsPhase])

            # compute the PD force at the 2nd point of the trapezoidal for all the knots of the phase
            posErrMat_next = (goalMat - objPos[:,prev_knotsPhase+1:knotsPhase+1])
            PD_next = K*posErrMat_next + D*(-objVel[:,prev_knotsPhase+1:knotsPhase+1])

            # for the not contact phase
            if prev_knotsPhase < handCntForceIdx[0] or knotsPhase > handCntForceIdx[1]:
                # build the constraints
                gDym = DynamTq3D(objVel[:,prev_knotsPhase:knotsPhase], objVel[:,prev_knotsPhase+1:knotsPhase+1], \
                             np.tile(np.zeros(3),[numknotsPhase, 1]).T, np.tile(np.zeros(3),[numknotsPhase, 1]).T, \
                             dtime[numPhase]/float(numknotsPhase),\
                             np.tile(self.grav,[numknotsPhase, 1]).T, \
                             PD_cur, PD_next )
            else: # for the contact phase
                # build the constraints
                gDym = DynamTq3D(objVel[:,prev_knotsPhase:knotsPhase], objVel[:,prev_knotsPhase+1:knotsPhase+1], \
                                 # in case of switching phase numknotsPhase = numknotsPhase.size2(), but not for mulitphase but all contact
                                 handCntForce[:,:numknotsPhase], handCntForce[:,1:numknotsPhase+1], \
                                 dtime[numPhase]/float(numknotsPhase),\
                                 np.tile(self.grav,[numknotsPhase, 1]).T, \
                                  PD_cur, PD_next )

            # index to end knot of the phase, which will be the first knot of the next phase
            prev_knotsPhase = knotsPhase

            # gDym.size1()*gDym.size2() are the number of bounds as many as the constraints
            # and zero [0] as the bound of the dynamics constraints has to be zero
            # lower bound
            self.g_boundList[0] += [0]*gDym.size1()*gDym.size2()
            # upper bound
            self.g_boundList[1] += [0]*gDym.size1()*gDym.size2()

            # build the constraint mat
            self.g = ca.vertcat(self.g, ca.reshape(gDym, gDym.size1()*gDym.size2(), 1))


    def addDynIntegConst2(self, name1, name2, goal, K, D):
        """
         Add dynamics constraints of the object motion, w.r.t to forces applied on the object
         by two end effectors
        """

        try:
            # get object related symbolic variables
            objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            objVel = self.varList[self.varIndexDict.get(self.dobjPosName)]
            dtime = self.varList[self.varIndexDict.get(self.dtname)]
            handCntForce1 = self.varList[self.varIndexDict.get("force" + name1)]
            handCntForce2 = self.varList[self.varIndexDict.get("force" + name2)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or time or contact hand force variables initialised \n')
            exit()

        # obtain the index in terms of knots in which these forces apply on
        # regarding the second hand
        handCntForceIdx_2 = self.varIndexDict.get("forceKnotIdx" + name2)

        # integrate velocities from forces for each phase
        prev_knotsPhase = 0
        for numPhase in range(self.numPh):
            numknotsPhase = self.CtrIntValpPhase[numPhase]      # number of knots in this phase
            knotsPhase = prev_knotsPhase + numknotsPhase        # index to end knot of the phase

            # aggregate dynamics function as many times as the number of knots in this phase
            DynamTq3D = con_Intgr.dynam_tq3D.map(numknotsPhase, "thread", self.threads)

            # create the goal matrices for the all the knots of the phase
            goalMat = ca.SX(np.tile(np.array([goal[0],goal[1],goal[2]]),[numknotsPhase, 1]).T)

            # compute the PD force at the 1st point of the trapezoidal for all the knots of the phase
            posErrMat_cur = ( goalMat - objPos[:,prev_knotsPhase:knotsPhase])
            PD_cur =  K*posErrMat_cur + D*(-objVel[:,prev_knotsPhase:knotsPhase])

            # compute the PD force at the 2nd point of the trapezoidal for all the knots of the phase
            posErrMat_next = (goalMat - objPos[:,prev_knotsPhase+1:knotsPhase+1])
            PD_next = K*posErrMat_next + D*(-objVel[:,prev_knotsPhase+1:knotsPhase+1])

            # for the not contact phase of the hand2
            if prev_knotsPhase < handCntForceIdx_2[0] or knotsPhase > handCntForceIdx_2[1]:
                # build the constraints
                gDym = DynamTq3D(objVel[:,prev_knotsPhase:knotsPhase], objVel[:,prev_knotsPhase+1:knotsPhase+1], \
                                 handCntForce1[:,prev_knotsPhase:knotsPhase], handCntForce1[:,prev_knotsPhase+1:knotsPhase+1],\
                                 dtime[numPhase]/float(numknotsPhase),\
                                 np.tile(self.grav,[numknotsPhase, 1]).T, \
                                 PD_cur, PD_next)

            else: # for the contact phase
                # build the constraints
                gDym = DynamTq3D(objVel[:,prev_knotsPhase:knotsPhase], objVel[:,prev_knotsPhase+1:knotsPhase+1], \
                                 handCntForce1[:,prev_knotsPhase:knotsPhase] + handCntForce2[:,prev_knotsPhase-handCntForceIdx_2[0]:knotsPhase-handCntForceIdx_2[0]]  ,\
                                 handCntForce1[:,prev_knotsPhase+1:knotsPhase+1] + handCntForce2[:,prev_knotsPhase+1-handCntForceIdx_2[0]:knotsPhase+1-handCntForceIdx_2[0]], \
                                 dtime[numPhase]/float(numknotsPhase),\
                                 np.tile(self.grav,[numknotsPhase, 1]).T, \
                                 PD_cur, PD_next)

            # index to end knot of the phase, which will be the first knot of the next phase
            prev_knotsPhase = knotsPhase

            # gDym.size1()*gDym.size2() are the number of bounds as many as the constraints
            # and zero [0] as the bound of the dynamics constraints has to be zero
            # lower bound
            self.g_boundList[0] += [0]*gDym.size1()*gDym.size2()
            # upper bound
            self.g_boundList[1] += [0]*gDym.size1()*gDym.size2()

            # build the constraint mat
            self.g = ca.vertcat(self.g, ca.reshape(gDym, gDym.size1()*gDym.size2(), 1))

        # ----------------------------------------------------------------------
        # single constraint, just for the transition from one phase to the other
        # ----------------------------------------------------------------------


        # check if the both phases are contact for the second hand
        if handCntForceIdx_2[0] == 0 and handCntForceIdx_2[1] == self.N:

            # # compute the PD force at the 1st point of the trapezoidal for all the knots of the phase
            posErrMat_cur = ( goalMat - objPos[:, self.CtrIntValpPhase[0]])
            PD_curInter =  K*posErrMat_cur + D*(-objVel[:, self.CtrIntValpPhase[0]])

            # compute the PD force at the 2nd point of the trapezoidal for all the knots of the phase
            posErrMat_next = (goalMat - objPos[:, self.CtrIntValpPhase[0]+1])
            PD_nextInter = K*posErrMat_next + D*(-objVel[:, self.CtrIntValpPhase[0]+1])

            gDymInter = con_Intgr.dynam_tq3D(objVel[:, self.CtrIntValpPhase[0]], objVel[:, self.CtrIntValpPhase[0]+1], \
                                             handCntForce1[:, self.CtrIntValpPhase[0]] + handCntForce2[:,self.CtrIntValpPhase[0]] ,\
                                             handCntForce1[:, self.CtrIntValpPhase[0]+1] + handCntForce2[:,self.CtrIntValpPhase[0]+1] , \
                                             dtime[1]/float(self.CtrIntValpPhase[1]),\
                                             self.grav.T, \
                                             PD_curInter[:,0], PD_nextInter[:,0])


        # check if the first phase is contact for the swing hand
        elif handCntForceIdx_2[0] == 0:

            # compute the PD force at the 1st point of the trapezoidal for all the knots of the phase
            posErrMat_cur = ( goalMat - objPos[:,handCntForceIdx_2[1]])
            PD_curInter =  K*posErrMat_cur + D*(-objVel[:,handCntForceIdx_2[1]])

            # compute the PD force at the 2nd point of the trapezoidal for all the knots of the phase
            posErrMat_next = (goalMat - objPos[:,handCntForceIdx_2[1]+1])
            PD_nextInter = K*posErrMat_next + D*(-objVel[:,handCntForceIdx_2[1]+1])

            gDymInter = con_Intgr.dynam_tq3D(objVel[:,handCntForceIdx_2[1]], objVel[:,handCntForceIdx_2[1]+1], \
                                             handCntForce1[:,handCntForceIdx_2[1]] + handCntForce2[:,handCntForceIdx_2[1]]  ,\
                                             handCntForce1[:,handCntForceIdx_2[1]+1] + 0, \
                                             dtime[1]/float(self.CtrIntValpPhase[1]),\
                                             self.grav.T, \
                                             PD_curInter[:,0], PD_nextInter[:,0])


        # check if the first phase is swing for the swing hand
        elif handCntForceIdx_2[0] > 0:

            # compute the PD force at the 1st point of the trapezoidal for all the knots of the phase
            posErrMat_cur = ( goalMat - objPos[:,handCntForceIdx_2[0]-1])
            PD_curInter =  K*posErrMat_cur + D*(-objVel[:,handCntForceIdx_2[0]-1])

            # compute the PD force at the 2nd point of the trapezoidal for all the knots of the phase
            posErrMat_next = (goalMat - objPos[:,handCntForceIdx_2[0]])
            PD_nextInter = K*posErrMat_next + D*(-objVel[:,handCntForceIdx_2[0]])

            gDymInter = con_Intgr.dynam_tq3D(objVel[:,handCntForceIdx_2[0]-1], objVel[:,handCntForceIdx_2[0]], \
                                             handCntForce1[:,handCntForceIdx_2[0]-1] + 0 ,\
                                             handCntForce1[:,handCntForceIdx_2[0]] + handCntForce2[:,0] , \
                                             dtime[1]/float(self.CtrIntValpPhase[1]),\
                                             self.grav.T, \
                                             PD_curInter[:,0], PD_nextInter[:,0])


        # lower bound
        self.g_boundList[0] += [0]*gDymInter.size1()*gDymInter.size2()
        # upper bound
        self.g_boundList[1] += [0]*gDymInter.size1()*gDymInter.size2()

        # append to the constraint mat the transition constraint from one phase to the other
        self.g = ca.vertcat(self.g, ca.reshape(gDymInter, gDymInter.size1()*gDymInter.size2(), 1))


    def addTorqueCntConst(self, name, cnt_fi=None):
        """
         Add torque constraints applied on the object w.r.t the respective force
        """

        try:
            # get object related symbolic variables
            objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            handCntForce = self.varList[self.varIndexDict.get("force" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or contact hand force variables initialised \n')
            exit()

        # obtain the index in terms of knots in which these forces apply on
        handCntForceIdx = self.varIndexDict.get("forceKnotIdx" + name)

        # aggregate torque function as many times as the number of knots of the contact end-effector
        TorqueCnt = con_Tau.param_torque_Obj.map(handCntForce.size2(),  "thread", self.threads)

        # build the constraints
        if cnt_fi is not None:
            gTorque = handCntForce[2,:] - TorqueCnt(cnt_fi, objPos[2, handCntForceIdx[0]:handCntForceIdx[-1]+1], self.Inertia, self.mass, handCntForce[:2,:])
        else:
            # get the contact location of the last swing knot (landing point )
            handSwPolar = self.varList[self.varIndexDict.get("polarPos" + name)]
            handSwCntIdx = self.varIndexDict.get("polarKnotIdx" + name)
            gTorque = handCntForce[2,:] - TorqueCnt(handSwPolar[0, handSwCntIdx[-1]-1], objPos[2, handCntForceIdx[0]:handCntForceIdx[-1]+1], self.Inertia, self.mass, handCntForce[:2,:])

        # gDym.size1()*gDym.size2() are the number of bounds as many as the constraints
        # and zero [0] as the bound of the torque constraints have to be zero
        # lower bound
        self.g_boundList[0] += [0]*gTorque.size1()*gTorque.size2()
        # upper bound
        self.g_boundList[1] += [0]*gTorque.size1()*gTorque.size2()

        # build the constraint mat
        self.g = ca.vertcat(self.g, ca.reshape(gTorque, gTorque.size1()*gTorque.size2(), 1))


    def addFrictionConeCntConst(self, name, cnt_fi=None):
        """
         Add friction cone constraints applied on the object for the forces
        """

        try:
            # get object related symbolic variables
            objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            handCntForce = self.varList[self.varIndexDict.get("force" + name)]
            handCntForceCoeffs = self.varList[self.varIndexDict.get("f_coefs" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or contact hand force or force coefficients variables initialised \n')
            exit()

        # obtain the index in terms of knots in which these forces apply on
        handCntForceIdx = self.varIndexDict.get("forceKnotIdx" + name)

        # aggregate friction cone function as many times as the number of knots of the contact end-effector
        FrConeCnt = con_Vfc.param_force.map(handCntForce.size2(),  "thread", self.threads)

        # build the constraints
        if cnt_fi is not None:
            gForceCoeff = handCntForce[:2,:] - FrConeCnt(cnt_fi, objPos[2, handCntForceIdx[0]:handCntForceIdx[-1]+1], self.muCone, handCntForceCoeffs[:,:])
        else:
            # get the contact location of the last swing knot (landing point )
            handSwPolar = self.varList[self.varIndexDict.get("polarPos" + name)]
            handSwCntIdx = self.varIndexDict.get("polarKnotIdx" + name)

            gForceCoeff = handCntForce[:2,:] - FrConeCnt(handSwPolar[0, handSwCntIdx[-1]-1], objPos[2, handCntForceIdx[0]:handCntForceIdx[-1]+1], self.muCone, handCntForceCoeffs[:,:])

        # gForceCoeff.size1()*gForceCoeff.size2() are the number of bounds as many as the constraints
        # and zero [0] as the bound of the friction cone constraints have to be zero
        # lower bound
        self.g_boundList[0] += [0]*gForceCoeff.size1()*gForceCoeff.size2()
        # upper bound
        self.g_boundList[1] += [0]*gForceCoeff.size1()*gForceCoeff.size2()

        # build the constraint mat
        self.g = ca.vertcat(self.g, ca.reshape(gForceCoeff, gForceCoeff.size1()*gForceCoeff.size2(), 1))


    def addSwingContConst(self, name, withCnt=False):
        """
         Add constraints that the end effector should not be in contact with the object
        """

        try:
            # get object related symbolic variables
            handSwPolar = self.varList[self.varIndexDict.get("polarPos" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have contact hand polar variables initialised \n')
            exit()

        # aggregate distance function as many times as the number of knots of the contact end-effector
        DistSw = obj_Dist_Norm_Cone.dist.map(handSwPolar.size2(),  "thread", self.threads)

        # build the constraints
        # polar fi is : handSwPolar[0,:]
        # polar radius is : handSwPolar[1,:]
        gSwDist = DistSw(handSwPolar[0,:], handSwPolar[1,:])

        # gSwDist.size1()*gSwDist.size2() are the number of bounds as many as the constraints
        # and zero [0] as the bound of the friction cone constraints have to be zero
        # lower bound
        self.g_boundList[0] += [self.maxSwDist]*gSwDist.size1()*gSwDist.size2()
        # upper bound
        self.g_boundList[1] += [0]*gSwDist.size1()*gSwDist.size2()

        # if last knot needs to be in contact (landing knot)
        if withCnt:
            # lower bound set to zero so in contact
            self.g_boundList[0][-1] = 0

        # build the constraint mat
        self.g = ca.vertcat(self.g, ca.reshape(gSwDist, gSwDist.size1()*gSwDist.size2(), 1))


    def addSwingLocalIntegConst(self, name, cnt_fi=None):
        """
         Add integration constraints of the swing end-effector object motion
        """
        try:
            # get object related symbolic variables
            dtime = self.varList[self.varIndexDict.get(self.dtname)]
            handSwPolar = self.varList[self.varIndexDict.get("polarPos" + name)]
            dhandSwPolar = self.varList[self.varIndexDict.get("polarVel" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or time or swing end effector variables initialised \n')
            exit()

        # aggregate integration function as many times as the number of knots in this phase
        TrapSw2D = con_Intgr.trap_quad2D.map(handSwPolar.size2()-1,  "thread", self.threads)

        # which means that the swing phase is second
        if cnt_fi is not None:
            phaseIdx = 1

            # build a constraint that enforces continuity from the current contact location to the 1st swing knot
            gInterCnt2Sw = con_Intgr.trap_quad2D( ca.vertcat(cnt_fi, obj_Dist_Norm_Cone.dist_spl(cnt_fi)), handSwPolar[:,0], \
                                                  np.zeros(2).T, dhandSwPolar[:,0], \
                                                  dtime[phaseIdx]/float(self.CtrIntValpPhase[phaseIdx]))

            # integrate position from velocities of the end effector
            # append to the contact2swing constraint all the swing2swing constraints
            gInterSw = TrapSw2D(handSwPolar[:,:handSwPolar.size2()-1], handSwPolar[:,1:handSwPolar.size2()], \
                                dhandSwPolar[:,:handSwPolar.size2()-1], dhandSwPolar[:,1:handSwPolar.size2()], \
                                dtime[phaseIdx]/float(self.CtrIntValpPhase[phaseIdx]))

            gInterSw = ca.horzcat(gInterCnt2Sw, gInterSw).T

        else: # which means that the swing phase is first
            phaseIdx = 0

            # integrate position from velocities of the end effector
            # append to the contact2swing constraint all the swing2swing constraints
            gInterSw = TrapSw2D(handSwPolar[:,:handSwPolar.size2()-1], handSwPolar[:,1:handSwPolar.size2()], \
                                dhandSwPolar[:,:handSwPolar.size2()-1], dhandSwPolar[:,1:handSwPolar.size2()], \
                                dtime[phaseIdx]/float(self.CtrIntValpPhase[phaseIdx]))

            # build a constraint that enforces continuity from the current contact location to the 1st swing knot
            gInterSw2Cnt = con_Intgr.trap_quad2D( handSwPolar[:,-1],  handSwPolar[:,-1],\
                                                  dhandSwPolar[:,-1], np.zeros(2).T, \
                                                  dtime[phaseIdx]/float(self.CtrIntValpPhase[phaseIdx]))

            gInterSw = ca.horzcat(gInterSw, gInterSw2Cnt).T

        # gInterSw.size1()*gInterSw.size2() are the number of bounds as many as the constraints
        # and zero [0] as the bound of the integration constraints has to be zero
        # lower bound
        self.g_boundList[0] += [0]*gInterSw.size1()*gInterSw.size2()
        # upper bound
        self.g_boundList[1] += [0]*gInterSw.size1()*gInterSw.size2()

        # build the constraint mat
        self.g = ca.vertcat(self.g, ca.reshape(gInterSw,gInterSw.size1()*gInterSw.size2(),1))


    # ATTENTION : Has not been implemented yet!!!!!!
    def addSwingGlobalIntegConst(self, cnt_fi, name):
        """
         Add integration constraints of the swing end-effector with object motion
        """
        try:
            # get object related symbolic variables
            # objPos = self.varList[self.varIndexDict.get(self.objPosName)]
            # objVel = self.varList[self.varIndexDict.get(self.dobjPosName)]
            dtime = self.varList[self.varIndexDict.get(self.dtname)]
            handSwPolar = self.varList[self.varIndexDict.get("polarPos" + name)]
            dhandSwPolar = self.varList[self.varIndexDict.get("polarVel" + name)]
        except AttributeError:
            print('\n \x1b[0;30;41m' + 'ERROR' + '\x1b[0m' + \
             ': Optimization problem does not have object or time or swing end effector variables initialised \n')
            exit()

        # obtain the index in terms of knots in which these swing motion happens
        # handSwPolarIdx = self.varIndexDict.get("polarKnotIdx" + name)

        #######    # TEMP >.................................
        numknotsPhase = self.CtrIntValpPhase[1]      # number of knots in this phase

        # print handSwPolarIdx , handSwPolar.size2(), cnt_fi
        # build a constraint that enforces continuity from the current contact location to the 1st swing knot
        gInterCnt2Sw = con_Intgr.trap_quad2D( ca.vertcat(cnt_fi, obj_Dist_Norm_Cone.dist_spl(cnt_fi)), handSwPolar[:,0], \
                                              np.zeros(2).T, dhandSwPolar[:,0], \
                                              dtime[1]/float(numknotsPhase))


        # aggregate integration function as many times as the number of knots in this phase
        TrapSw2D = con_Intgr.trap_quad2D.map(handSwPolar.size2()-1, "thread", self.threads)

        # integrate position from velocities for each phase
        gInterSw = ca.vertcat(gInterCnt2Sw.T, \
                    TrapSw2D(handSwPolar[:,:handSwPolar.size2()-1], handSwPolar[:,1:handSwPolar.size2()], \
                            dhandSwPolar[:,:handSwPolar.size2()-1], dhandSwPolar[:,1:handSwPolar.size2()], \
                            dtime[1]/float(numknotsPhase)))

        # gInterSw = TrapSw2D(objPos[:,handSwPolarIdx[0]:handSwPolarIdx[-1]-1], objPos[:,handSwPolarIdx[0]+1:handSwPolarIdx[-1]], \
        #                     objVel[:,handSwPolarIdx[0]:handSwPolarIdx[-1]-1], objVel[:,handSwPolarIdx[0]+1:handSwPolarIdx[-1]], \
        #                     dtime[1]/float(numknotsPhase))


        # gInterSw.size1()*gInterSw.size2() are the number of bounds as many as the constraints
        # and zero [0] as the bound of the integration constraints has to be zero
        # lower bound
        self.g_boundList[0] += [0]*gInterSw.size1()*gInterSw.size2()
        # upper bound
        self.g_boundList[1] += [0]*gInterSw.size1()*gInterSw.size2()

        # build the constraint mat
        self.g = ca.vertcat(self.g, ca.reshape(gInterSw,gInterSw.size1()*gInterSw.size2(),1))

        # prints for debugging reasons
        # print "Size of g constraint mat" ,self.g.size1() , " ",self.g.size2()

        # prints for debugging reasons
        # print "Size of g constraint mat" ,self.g.size1() , " ",self.g.size2()


    ### -------------------------------------------------------------------------###
    ###                       Functions for bound building                       ###
    ### -------------------------------------------------------------------------###


    def buildPosVelBoundFun(self):
        '''
        Constructs a CasADi function to generate bound vector for position and
        velocity of the object
        '''

        # retrieve variables related for object
        pObj, dpObj = self.getObjVars()

        # variables to build position bounds
        initPos = ca.SX.sym("initPos",3)
        minPos = ca.SX.sym("minPos",3)
        maxPos = ca.SX.sym("maxPos",3)

        Xlbx = [[initPos[0], initPos[1], initPos[2]]] + [[minPos[0], minPos[1], minPos[2]]]* (pObj.size2()-1)
        Xlbx = ca.reshape(np.array(Xlbx).T, 1, pObj.size2()*pObj.size1() )

        Xubx = [[initPos[0], initPos[1], initPos[2]]] + [[maxPos[0], maxPos[1], maxPos[2]]]* (pObj.size2()-1)
        Xubx = ca.reshape(np.array(Xubx).T, 1, pObj.size2()*pObj.size1() )

        # build the function for the position bounds
        bPos_f = ca.Function('bPos_f',[initPos, minPos, maxPos], [Xlbx, Xubx])

        # variables to build velocity bounds
        initVel = ca.SX.sym("initVel",3)
        minVel = ca.SX.sym("minVel",3)
        maxVel = ca.SX.sym("maxVel",3)

        dXlbx = [[initVel[0], initVel[1], initVel[2]]] + [[minVel[0], minVel[1], minVel[2]]]* (pObj.size2()-1)
        dXlbx = ca.reshape(np.array(dXlbx).T, 1, dpObj.size2()*dpObj.size1() )

        dXubx = [[initVel[0], initVel[1], initVel[2]]] + [[maxVel[0], maxVel[1], maxVel[2]]]* (pObj.size2()-1)
        dXubx = ca.reshape(np.array(dXubx).T, 1, dpObj.size2()*dpObj.size1() )

        # build the function for the position bounds
        bVel_f = ca.Function('bVel_f',[initVel, minVel, maxVel], [dXlbx, dXubx])

        return bPos_f, bVel_f


    def buildForceBoundFun(self, name):
        '''
        Constructs a CasADi function to generate bound vector for force of the hands
        '''

        # retrieve variables related to force of the contact end-effector
        forceCnt, forceCoefCnt = self.getContactEndEffVars(name)

        # variables to build force bounds
        bForce = ca.SX.sym("bForce"+name)
        bForCoeff = ca.SX.sym("bForCoeff"+name)

        # force hand bounds
        fCntlbx = ca.SX(np.array([-bForce]*forceCnt.size1()*forceCnt.size2())).T # transpose to be 1 row
        fCntubx = ca.SX(np.array([bForce]*forceCnt.size1()*forceCnt.size2())).T  # transpose to be 1 row

        # force coef hand bounds
        fcoefCntlbx = ca.SX(np.array([0]*forceCoefCnt.size1()*forceCoefCnt.size2())).T # transpose to be 1 row
        fcoefCntubx = ca.SX(np.array([bForCoeff]*forceCoefCnt.size1()*forceCoefCnt.size2())).T # transpose to be 1 row

        # build the function for the cnt force bounds
        bForceCnt_f = ca.Function('bForceCnt_f'+name,[bForce, bForCoeff], [fCntlbx, fCntubx, fcoefCntlbx, fcoefCntubx])

        return bForceCnt_f


    def buildSwingBoundFun(self, name, SwCnt=False, CntSw=False):
        '''
        Constructs a CasADi function to generate bound vector for swing hand position
        '''

        # retrieve variables related to force of the contact end-effector
        _, _, BpolarPosSw, BpolarVelSw = self.getContactSwingEndEff(name)

        # variables to build polar bounds
        minPolarTheta = ca.SX.sym("minPolarTheta")
        minPolarRadius = ca.SX.sym("minPolarRadius")
        maxPolarTheta = ca.SX.sym("maxPolarTheta")
        maxPolarRadius = ca.SX.sym("maxPolarRadius")

        # initial pose(polar coord) of swing hand
        initPolarTheta = ca.SX.sym("initPolarTheta")
        initPolarRadius = ca.SX.sym("initPolarRadius")

        # final pose(polar coord) of swing hand
        finalPolarTheta = ca.SX.sym("finalPolarTheta")
        finalPolarRadius = ca.SX.sym("finalPolarRadius")

        # variables to build velocity bounds
        maxd_PolarTheta = ca.SX.sym("maxd_PolarTheta")
        maxd_PolardRadius = ca.SX.sym("maxd_PolardRadius")


        if SwCnt:
            # position hand bounds
            BpolarPoslbx = [[initPolarTheta, initPolarRadius]] + [[minPolarTheta, minPolarRadius]]*(BpolarPosSw.size2()-2) + [[finalPolarTheta - 0.10, minPolarRadius]]
            BpolarPoslbx = ca.reshape(np.array(BpolarPoslbx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            BpolarPosubx = [[initPolarTheta, initPolarRadius]] + [[maxPolarTheta, maxPolarRadius]]*(BpolarPosSw.size2()-2) + [[finalPolarTheta + 0.10, maxPolarRadius]]
            BpolarPosubx = ca.reshape(np.array(BpolarPosubx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            # build the function for the swing position bounds
            bPosSw_f = ca.Function('bPosSw_f',[initPolarTheta, initPolarRadius, minPolarTheta, minPolarRadius, maxPolarTheta, maxPolarRadius, finalPolarTheta], \
                                             [BpolarPoslbx, BpolarPosubx])

            # velocity hand bounds
            BpolarVellbx = [[-maxd_PolarTheta, -maxd_PolardRadius]]*(BpolarPosSw.size2()-1) + [[-0.1*maxd_PolarTheta,  -0.1*maxd_PolardRadius]]
            BpolarVellbx = ca.reshape(np.array(BpolarVellbx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            BpolarVelubx = [[maxd_PolarTheta, maxd_PolardRadius]]*(BpolarPosSw.size2()-1) + [[0.1*maxd_PolarTheta,  0.1*maxd_PolardRadius]]
            BpolarVelubx = ca.reshape(np.array(BpolarVelubx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            # build the function for the swing velocity bounds
            bVelSw_f = ca.Function('bVelSw_f',[maxd_PolarTheta, maxd_PolardRadius], \
                                              [BpolarVellbx, BpolarVelubx])


        if CntSw:
            # position hand bounds
            BpolarPoslbx = [[minPolarTheta, minPolarRadius]]*(BpolarPosSw.size2()-1) + [[finalPolarTheta - 0.35, finalPolarRadius - 0.2]]
            BpolarPoslbx = ca.reshape(np.array(BpolarPoslbx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            BpolarPosubx = [[maxPolarTheta, maxPolarRadius]]*(BpolarPosSw.size2()-1) + [[finalPolarTheta + 0.35, finalPolarRadius + 0.2]]
            BpolarPosubx = ca.reshape(np.array(BpolarPosubx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            bPosSw_f = ca.Function('bPosSw_f',[finalPolarTheta, finalPolarRadius, minPolarTheta, minPolarRadius, maxPolarTheta, maxPolarRadius], \
                                              [BpolarPoslbx, BpolarPosubx])


            # velocity hand bounds
            BpolarVellbx = [[-maxd_PolarTheta, -maxd_PolardRadius]]*(BpolarPosSw.size2())
            BpolarVellbx = ca.reshape(np.array(BpolarVellbx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            BpolarVelubx = [[maxd_PolarTheta, maxd_PolardRadius]]*(BpolarPosSw.size2())
            BpolarVelubx = ca.reshape(np.array(BpolarVelubx).T, 1, BpolarPosSw.size1()*(BpolarPosSw.size2()))

            # build the function for the swing velocity bounds
            bVelSw_f = ca.Function('bVelSw_f',[maxd_PolarTheta, maxd_PolardRadius], \
                                              [BpolarVellbx, BpolarVelubx])


        return bPosSw_f, bVelSw_f


    def buildConstraintBounds(self):
        '''
        Constructs a CasADi function to generate bound vector for the constraints
        '''

        # get the bounds of the constraints
        _, gBounds = self.getConstraints()

        lbg = ca.SX(gBounds[0,:]).T
        ubg = ca.SX(gBounds[1,:]).T

        none = ca.SX.sym("none")

        bg_f = ca.Function('bg_f',[none],[lbg, ubg])

        return bg_f

    ### -------------------------------------------------------------------------###
    ###                       Functions for Solution decoding                    ###
    ### -------------------------------------------------------------------------###

    def decodeSolution(self, nameA, nameB, SwCnt=False, CntSw=False):
        '''
        Constructs CasADi functions that decode the solution vector to meaningful quants
        '''

        # build a hypothetical SX that has the size of the solution vector
        Solution = ca.SX.sym("Solution", self.varVect.size1())

        ### ------------------------- Position-------------------------------###
        pos = Solution[self.varVectIndexDict.get('objPos')[0]:self.varVectIndexDict.get('objPos')[-1]]
        pos3D = ca.reshape(pos, 3, int(pos.size1()/3))
        getPos = ca.Function('getPos',[Solution],[pos3D])

        ### ------------------------- Velocity-------------------------------###
        vel = Solution[self.varVectIndexDict.get('objVel')[0]:self.varVectIndexDict.get('objVel')[-1]]
        vel3D = ca.reshape(vel, 3, int(vel.size1()/3))
        getVel = ca.Function('getVel',[Solution],[vel3D])

        ### ------------------------- Time ----------------------------------###
        # access the time variables
        t1 = Solution[self.varVectIndexDict.get('dt')[0]]
        t2 = Solution[self.varVectIndexDict.get('dt')[-1]-1]

        timeVec = ca.SX.ones(1, self.N+1)
        timeVec[0] = 0
        timeVec[1:self.CtrIntValpPhase[0]+1] = t1/self.CtrIntValpPhase[0]
        timeVec[self.CtrIntValpPhase[0]+1:self.CtrIntValpPhase[0]+self.CtrIntValpPhase[-1]+1] = t2/self.CtrIntValpPhase[-1]

        # could be written with the function fold for efficiency at construction
        x = 0
        for i in range(self.N+1):
            x = timeVec[i] + x
            timeVec[i] = x


        getTimes = ca.Function('getTimes',[Solution],[t1, t2, timeVec.T])


        ### ------------------------- Forces eeA ----------------------------###
        Aforces = Solution[self.varVectIndexDict.get("force" + nameA)[0]:self.varVectIndexDict.get("force" + nameA)[-1]]
        Aforces3D = ca.reshape(Aforces, 3, int(Aforces.size1()/3))

        getForcesA = ca.Function('getForcesA',[Solution],[Aforces3D])

        ### ------------------------- Forces eeB ----------------------------###
        Bforces = Solution[self.varVectIndexDict.get("force" + nameB)[0]:self.varVectIndexDict.get("force" + nameB)[-1]]
        Bforces3D = ca.reshape(Bforces, 3, int(Bforces.size1()/3))

        if SwCnt:
            # append force vector with zeros for the swing phase of the end effector
            Bforces3D = ca.horzcat( ca.SX.zeros(3, int(pos.size1()/3) - int(Bforces.size1()/3)), Bforces3D )
        elif CntSw:
            # append force vector with zeros for the swing phase of the end effector
            Bforces3D = ca.horzcat( Bforces3D, ca.SX.zeros(3, int(pos.size1()/3) - int(Bforces.size1()/3)))

        getForcesB = ca.Function('getForcesB',[Solution],[Bforces3D])


        ### ------------------------- Polar eeA ----------------------------###
        p_polar = ca.SX.sym("p_polar")

        cart =  ca.DM.ones(2, int(pos.size1()/3))*obj_Dist_Norm_Cone.xy_spl(p_polar).T

        # define function that computes the relative cartesian coordinates of the cnt hand
        getCartA = ca.Function('getCartA',[p_polar],[cart])

        # create a function that computes normals for all knots of the problem
        NnomalFunc = obj_Dist_Norm_Cone.normal.map(int(pos.size1()/3),  "thread", self.threads)
        # rotate normal with to align with the Rcs format
        polarOrient = ca.mtimes(operators.R(-ca.pi/2), NnomalFunc(ca.DM.ones(int(pos.size1()/3))*p_polar, 0))
        # polarOrient = NnomalFunc(ca.DM.ones(pos.size1()/3)*p_polar, 0)


        # define function that computes the orienation of the cnt hand
        getPolarOrientA = ca.Function('getPolarOrientA',[p_polar],[ p_polar * ca.SX.ones(1, int(pos.size1()/3)) ])

        if (SwCnt == False and CntSw == False):
            # hand B is also a contact hand, thus the same functions apply
            getCartB = ca.Function('getCartB',[p_polar],[cart])
            # getPolarOrientB = ca.Function('getPolarOrientB',[p_polar],[ ca.atan2( polarOrient[1,:], polarOrient[0,:])* ca.SX.ones(1, pos.size1()/3)  ])
            getPolarOrientB = ca.Function('getPolarOrientB',[p_polar],[  p_polar * ca.SX.ones(1, int(pos.size1()/3)) ])


        else:
            ### ------------------------- Polar eeB ----------------------------###
            Bpolar = Solution[self.varVectIndexDict.get("polarPos" + nameB)[0]:self.varVectIndexDict.get("polarPos" + nameB)[-1]]

            # obtain polar coord
            Bpolar2D = ca.reshape(Bpolar, 2, int(Bpolar.size1()/2))

            # based on circle model (swing motion)
            Npolar2cartFuncCir = operators.circlePt.map(Bpolar2D.size2(), "thread", self.threads)


            # if in Swing and then in contact with the object
            if SwCnt:
                # cartesian coordinates of hand during swing
                Bcart2DSw  = Npolar2cartFuncCir(Bpolar2D[1,:], Bpolar2D[0,:])

                # cartesian coordinates of hand at landing and contact
                Bcart2DCnt = obj_Dist_Norm_Cone.xy_spl(Bpolar2D[0,-1]).T

                # rotate normal with to align with the Rcs format - ATTENTION
                BpolarOrientCnt =  obj_Dist_Norm_Cone.normal(Bpolar2D[0,-1], 0)

                Bcart2D = ca.horzcat( Bcart2DSw, Bcart2DCnt*ca.SX.ones(2, int(pos.size1()/3) - Bpolar2D.size2()) )
                # orientation of the object pos3D[2,i]

                BpolarOrient = ca.horzcat( Bpolar2D[0,:], Bpolar2D[0,-1]*ca.SX.ones(1, int(pos.size1()/3) - Bpolar2D.size2()))


            elif CntSw:   # if in contact with the object and then in swing motion

                # cartesian coordinates of hand at landing and contact
                Bcart2DCnt = obj_Dist_Norm_Cone.xy_spl(p_polar).T

                # rotate normal to align with the Rcs format - not used in python visualisation
                BpolarOrientCnt = obj_Dist_Norm_Cone.normal(p_polar, 0)

                # cartesian coordinates of hand during swing
                Bcart2DSw  = Npolar2cartFuncCir(Bpolar2D[1,:], Bpolar2D[0,:])

                Bcart2D = ca.horzcat( Bcart2DCnt*ca.SX.ones(2, int(pos.size1()/3) - Bpolar2D.size2()), Bcart2DSw )
                        # orientation of the object pos3D[2,i]

                BpolarOrient = ca.horzcat(  p_polar*ca.SX.ones(1, int(pos.size1()/3) - Bpolar2D.size2()), Bpolar2D[0,:])


            # define function that computes the cartesian location of the swing hand
            getCartB = ca.Function('getCartB',[Solution, p_polar],[Bcart2D])

            # define function that computes the orienation of the swing hand
            getPolarOrientB = ca.Function('getPolarOrientB',[Solution, p_polar],[ BpolarOrient ])


        return getPos, getVel, getTimes, getForcesA, getForcesB, getCartA, getPolarOrientA, getCartB, getPolarOrientB


# end of file
