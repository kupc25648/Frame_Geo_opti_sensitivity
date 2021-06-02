'''
=====================================================================
Master file for Frame structure shape optimization using sensitivity analysis and numerical optimization
=====================================================================
'''

#====================================================================
#Import Part
#====================================================================
from FEM_frame import *
from frame_GEN import *

#====================================================================
#Parameter Part
#====================================================================
num_x = 5 # Nodes in x direction(horizontal)
num_z = 5 # Nodes in z direction(horizontal)
span  = 1 # structural span length(m.)
diameter = 0.1 # structural element diameter(m.)
loadx = 0 # nodal load in x direction (N)
loady = -1000 # nodal load in y direction (N)
loadz = 0 # nodal load in z direction (N)
Young = 10000000000 # structural elements' young modulus(N/m2)
# -------------------------------------------------------------------
# Parameter for initializing the structure part
# nodal height(y) is parameterized by following function of (x and z)
# Y = (c1*(x**2) + c2*(x*y) + c3*(y**2) + c4*x + c5*y + c6 ) * c7
c1= 0.3
c2= 0.3
c3= 0.1
c4= 0.3
c5= 0.2
c6= 0.2
c7= -0.3
# -------------------------------------------------------------------
# forgame defines what kind of game(See detail in frame_GEN.py)
# If forgame is not None, ignore parameter for initializing the structure part
forgame= 1001
# -------------------------------------------------------------------
game_max_y_val= 1 # Maximum nodal height(m.)(vertical)
game_alpha =0.1 # adjust ment value * does not used in this program
brace = None # None for no braced grid structure, 1 for braced grid structure
selfweight=False # False for no self-weight * does not used in this program

# -------------------------------------------------------------------
# optimization parameter part
max_iter = 1000 # number of maximum iteration process
rate = 1 # rate of adjustment

#====================================================================
# Main program (optimization part)
#====================================================================
model_X = gen_model(num_x,num_z,span,diameter,loadx,loady,loadz,Young,c1,c2,c3,c4,c5,c6,c7,forgame,game_max_y_val,game_alpha,brace,selfweight)
dU_dy, dU_node = model_X.model.sensitivity(model_X.model.nodes[0])

# sensitivity optimization
small_vall = 10e-6
prev = model_X.model.U_full[0]
now = model_X.model.U_full[0]+1
init = model_X.model.U_full[0]
final = model_X.model.U_full[0]
num_iter = 0
for i in range(max_iter):
    for nod in range(len(model_X.model.nodes)):
        if model_X.model.nodes[nod].res[1] == 0:
            dU_dy, dU_node = model_X.model.sensitivity(model_X.model.nodes[nod])

            avg_0 = max(dU_node[:,0])
            avg_1 = max(dU_node[:,1])
            avg_2 = max(dU_node[:,2])
            avg_3 = max(dU_node[:,3])
            avg_4 = max(dU_node[:,4])
            avg_5 = max(dU_node[:,5])
            avg = [avg_0,avg_1,avg_2,avg_3,avg_4,avg_5]
            avg = max(avg)
            if (model_X.model.nodes[nod].coord[1] + rate*avg >= 0) and (model_X.model.nodes[nod].coord[1] + rate*avg <= 1):
                model_X.model.nodes[nod].coord[1] += rate*avg

            else:
                pass
            model_X.model.restore()
            model_X.model.gen_all()
            final = model_X.model.U_full[0]
            now = final
            num_iter += 1
        else:
            pass
    print('iteration {} | Strain Energy {}'.format(i,final))
    if now>=prev:
        break
    else:
        prev=now
model_X.savetxt('Sensitivity Optimization Solution.txt')
model_X.render('Sensitivity Optimization',num_iter,init,final)
