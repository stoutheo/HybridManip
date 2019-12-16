#-----------------------------------------------------------------------------#
# This file provides the basic tools to generate animations of the motion plan
#   ......
#-----------------------------------------------------------------------------#

# numpy
from py_pack import np
# yaml
import yaml

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation


# package utils
from py_pack import operators as oper


# plotting box
# source : https://stackoverflow.com/questions/31921313/matplotlib-animation-moving-square

# plotting arrow
# For more detail, see
# https://brushingupscience.wordpress.com/2016/06/21/matplotlib-animations-the-easy-way/


def animationfunc1Hand(x,y,yaw, cntX, cntY , U, V, dt, gX, gY, gPhi):

    # specifications of the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.grid()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # generate patch that represents a rectangle
    patch = patches.Rectangle((0, 0), 0, 0, fc='y')
    patch2 = patches.Rectangle((0, 0), 0, 0, fc='r')

    # generate patch that represents a contact location
    patch3 = plt.Circle((0, 0), 0, fc='c')

    # instantiate vector to visualise force
    qax = ax.quiver(0, 0, 0, 0, scale=100)

    goalV = np.dot(np.array(oper.R(gPhi)), np.array([0,10]))
    qGoal = ax.quiver(gX, gY, goalV[0], goalV[1], scale=100, color='g')
    goalV = np.dot(np.array(oper.R(gPhi)), np.array([10,0]))
    qGoal = ax.quiver(gX, gY, goalV[0], goalV[1], scale=100, color='b')

    def init():
        ax.add_patch(patch)
        ax.add_patch(patch2)
        ax.add_patch(patch3)

        return patch, patch2

    def animate(i):

        if params['object']['shape']['shapeID'] == 1:
            h = params['object']['shape']['rectangle']['height']
            w = params['object']['shape']['rectangle']['width']

        if params['object']['shape']['shapeID'] == 2:
            h = params['object']['shape']['parallelogram']['height']
            w = params['object']['shape']['parallelogram']['width']

        # main rectangle
        patch.set_width(w)
        patch.set_height(h)
        d_trans = np.dot(oper.R(yaw[i]),[w/2, h/2])
        patch.set_xy([x[i] - d_trans[0], y[i] - d_trans[1]])
        patch._angle = np.rad2deg(yaw[i])

        # secondary rectangle to denote orientation
        patch2.set_width(w/2)
        patch2.set_height(h/2)
        d_trans2 = np.dot(oper.R(yaw[i]),[w/2,0])
        patch2.set_xy([x[i] - d_trans2[0], y[i] - d_trans2[1]])
        patch2._angle = np.rad2deg(yaw[i])

        # location of the circle
        patch3.set_radius(0.025)
        patch3.center = (cntX[i], cntY[i])

        # update location(origin) of the vector
        qax.set_offsets((cntX[i], cntY[i]))
        # update orienation of the vector
        qax.set_UVC(U[i], V[i])

        return patch, patch2, patch3, qax

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=len(x),
                                   interval=dt*1000 + 100,
                                   blit=True,
                                   repeat=True)
    plt.show()


def animationfunc2Hands(x,y,yaw, cntX1, cntY1 , U1, V1, cntX2, cntY2 , U2, V2, dt,  gX, gY, gPhi):

    with open("py_pack/config/parameters.yml", 'r') as ymlfile:
        params = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # specifications of the figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.grid()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    if params['object']['shape']['shapeID'] == 0 :
        # generate patch that represents a circle/disk
        patch = plt.Circle((0, 0), 0, fc='y')
        patch2 = patches.Rectangle((0, 0), 0, 0, fc='r')


    if params['object']['shape']['shapeID'] == 1 or params['object']['shape']['shapeID'] == 2:
        # generate patch that represents a rectangle
        patch = patches.Rectangle((0, 0), 0, 0, fc='y')
        patch2 = patches.Rectangle((0, 0), 0, 0, fc='r')


    # generate patch that represents a contact location
    patch3 = plt.Circle((0, 0), 0, fc='c')

    # instantiate vector to visualise force
    qax1 = ax.quiver(0, 0, 0, 0, scale=100)

    # generate patch that represents a contact location
    patch4 = plt.Circle((0, 0), 0, fc='m')

    # instantiate vector to visualise force
    qax2 = ax.quiver(0, 0, 0, 0, scale=100)

    goalV = np.dot(np.array(oper.R(gPhi)), np.array([0,10]))
    qGoal = ax.quiver(gX, gY, goalV[0], goalV[1], scale=100, color=(0.5,0,0))
    goalV = np.dot(np.array(oper.R(gPhi)), np.array([10,0]))
    qGoal = ax.quiver(gX, gY, goalV[0], goalV[1], scale=100, color=(0,0.5,0))

    # show coordinate system of origin
    qCoordX = ax.quiver(-1, -1, 10, 0, scale=100, color=(1,0,0))
    qCoordY = ax.quiver(-1, -1, 0,  10, scale=100, color=(0,1,0))

    def init():
        ax.add_patch(patch)
        ax.add_patch(patch2)
        ax.add_patch(patch3)
        ax.add_patch(patch4)

        return patch, patch2, patch3, patch4

    def animate(i):

        if params['object']['shape']['shapeID'] == 0 :
            r = params['object']['shape']['circle']['radius']
            patch.set_radius(r)
            patch.center = (x[i], y[i])

            # secondary rectangle to denote orientation
            patch2.set_width(0.05)
            patch2.set_height(r)
            d_trans2 = np.dot(oper.R(yaw[i]),[0.025, 0])
            patch2.set_xy([x[i] - d_trans2[0], y[i] - d_trans2[1]])
            if (int(matplotlib.__version__[0])) < 2:
                patch2._angle = np.rad2deg(yaw[i])
            else:
                patch2.angle  = np.rad2deg(yaw[i])

        if params['object']['shape']['shapeID'] == 1:
            h = params['object']['shape']['rectangle']['height']
            w = params['object']['shape']['rectangle']['width']

        if params['object']['shape']['shapeID'] == 2:
            h = params['object']['shape']['parallelogram']['height']
            w = params['object']['shape']['parallelogram']['width']


        if params['object']['shape']['shapeID'] == 1 or params['object']['shape']['shapeID'] == 2:
            # main rectangle
            patch.set_width(w)
            patch.set_height(h)
            d_trans = np.dot(oper.R(yaw[i]),[w/2, h/2])
            patch.set_xy([x[i] - d_trans[0], y[i] - d_trans[1]])
            if (int(matplotlib.__version__[0])) < 2:
                patch._angle = np.rad2deg(yaw[i])
            else:
                patch.angle  = np.rad2deg(yaw[i])

            # secondary rectangle to denote orientation
            patch2.set_width(w/2)
            patch2.set_height(h/2)
            d_trans2 = np.dot(oper.R(yaw[i]),[w/2,0])
            patch2.set_xy([x[i] - d_trans2[0], y[i] - d_trans2[1]])
            if (int(matplotlib.__version__[0])) < 2:
                patch2._angle = np.rad2deg(yaw[i])
            else:
                patch2.angle  = np.rad2deg(yaw[i])

        # location of the circle
        patch3.set_radius(0.025)
        patch3.center = (cntX1[i], cntY1[i])

        # update location(origin) of the vector
        qax1.set_offsets((cntX1[i], cntY1[i]))
        # update orienation of the vector
        qax1.set_UVC(U1[i], V1[i])

        # location of the circle
        patch4.set_radius(0.025)
        patch4.center = (cntX2[i], cntY2[i])

        # update location(origin) of the vector
        qax2.set_offsets((cntX2[i], cntY2[i]))
        # update orienation of the vector
        qax2.set_UVC(U2[i], V2[i])


        return patch, patch2, patch3, patch4, qax1, qax2

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=len(x),
                                   interval=dt*1000 + 100,
                                   blit=True,
                                   repeat=True)
    plt.show()










# end of file
