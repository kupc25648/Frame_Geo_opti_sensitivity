'''
==================================================================
Space frame generate file
This file generate frame structure using parameters as input.
Uncomment 'Test' at the end of file to see result
Adjustable parameter are under '研究室'

スペースフレーム生成ファイル
このファイルは、入力としてパラメーターを使用してフレーム構造を生成します。
ファイルの最後で「テスト」のコメントを外して結果を確認します
調整可能なパラメータは「研究室」の下にあります

Acknowledgements
Forgame 1,2,3,4,5 were developed by
川上 梨鈴,
遠藤 凌也,
吉房 謙祥
,4th year students from Professor Yamamoto's lab 2020, Tokai University,
Department of Architecture

謝辞
Forgame 1,2,3,4,5は、東海大学山本憲司の研究室 2020
東海大学建築学科4年生の
川上 梨鈴,
遠藤 凌也,
吉房 謙祥
によって開発されました。

==================================================================
'''

'''
====================================================================
Import part
インポート部
====================================================================
'''
from FEM_frame import *
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
#import matplotlib as mpl # use for mac bigsur
#mpl.use('tkagg') # use for mac bigsur
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
import shutil
import csv
import ast
from os import listdir
from os.path import isfile, join

#--------------------------------
# Frame generate class / フレーム生成クラス
#--------------------------------
class gen_model:
    def __init__(self,
                num_x
                ,num_z
                ,span
                ,diameter
                ,loadx
                ,loady
                ,loadz
                ,Young
                ,c1,c2,c3,c4,c5,c6,c7,forgame=None,game_max_y_val=None,game_alpha=0.1,brace=None,selfweight=False):

        # Frame Dimensions / フレームの寸法
        self.num_x = num_x # number of node in x direction (horizontal) / x方向のノード数（水平）
        self.num_z = num_z # number of node in z direction (horizontal) / z方向のノード数（水平）
        self.span = span # span's length (m.) / スパンの長さ（m。）
        self.dia = diameter # element diameter (m.) / 要素の直径（m。）
        self.area = np.pi*(self.dia**2)/4 # element area (m2) / 要素面積（m2）
        self.ival = np.pi*((self.dia)**4)/64 # element moment of inertia (m4) / 要素慣性モーメント（m4）
        self.jval = np.pi*((self.dia)**4)/32 # element polar moment of inertia (m4) / 要素極慣性モーメント（m4）
        self.brace = brace # structure have brace element (boolean yes/no) / 構造にはブレース要素があります（ブール値yes / no）
        # -------------------------------------------------
        # Frame Load / 荷重
        self.loadx = loadx # load in x direction(N) / x方向の荷重（N）
        self.loady = loady # load in y direction(N)-vertical / y方向の荷重（N）-垂直
        self.loadz = loadz # load in z direction(N) / z方向の荷重（N）
        # -------------------------------------------------
        # Frame Properties / フレームのプロパティ
        self.Young = Young # element Young modulus(N/m2) / 要素ヤング率（N / m2）
        self.shearmodulus = Young/2 # element Shear modulus(N/m2) / 要素のせん断弾性率（N / m2）
        # -------------------------------------------------
        # lists used to store structural data / 構造データを格納するために使用されるリスト
        self.n_u_x = [] # value of x coordinate / x座標の値
        self.n_u_z = [] # value of z coordinate / z座標の値
        self.n_u_coord = [] # value of [x,y,z] coordinate / [x、y、z]座標の値
        self.n_u_name_div =[] #node of the structure / 構造のノード
        # -------------------------------------------------
        # Parameter for y coordinate value function y座標値関数のパラメーター
        # Y = (c1*(x**2) + c2*(x*y) + c3*(y**2) + c4*x + c5*y + c6 ) * c7
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.c7 = c7
        # -------------------------------------------------
        # Reinforcement Learning Defaults / 強化学習のデフォルト
        # 0: minimal surface game / 最小限の表面ゲーム
        # 1: structural generation game / 構造生成ゲーム
        # 2: structural optimization game / 構造最適化ゲーム
        self.forgame = forgame
        # Maximum y coordinate(m.) / 最大y座標（m。）
        self.game_max_y_val = game_max_y_val
        # -------------------------------------------------
        # Frame Model (will be generated after init) / フレームモデル（初期化後に生成されます）
        self.model = None
        # -------------------------------------------------
        # Frame Surface (will be generated after init) / フレームモデル（初期化後に生成されます）
        self.surface_1 = 0 #3d triangle method
        self.surface_2 = 0 #shape functions method
        # -------------------------------------------------
        # Initial Ramdom Values
        self.alpha = game_alpha
        # -------------------------------------------------
        # self_weight (N/m)
        self.use_selfweight=selfweight
        self.weight = 50
        # -------------------------------------------------
        # Generate functions / 関数を生成する
        self.gennode()
        self.generate()

    # Function to perform Global Decomposition of y deformation
    # There is a bug in this function: please check Q1,Q2,Q3,Q4
    def Decomp_dy(self,n_u_name_div,agent_position,lz=None,lx=None):
        agent_position = [2,2]
        '''
        1. Create base decom matrix
        2. Divide matrix into 4 parts using agent_position(z,x) as central point
        3. In each matrix part,
            3.1 Create a list of [position, y deformation]
            3.2 Sort list in 3.1 from furthest from the agent_position to closest
            3.3 Perform Global Decomposition y deformation
            3.4 Assign [Global Decomposition y deformation] back into decom matrix using [position]
        4. Return decom matrix
        '''
        decom_dy = np.zeros((len(n_u_name_div),len(n_u_name_div[0])))
        '''
        for i in range(len(self.model.nodes)):
            print("position x:{}, z:{}".format(self.model.nodes[i].coord[0]+1,self.model.nodes[i].coord[2]+1))
        print('-------')
        for i in range(len(n_u_name_div[0])):
            print("position x:{}, z:{}".format(n_u_name_div[0][i].coord[0]+1,n_u_name_div[0][i].coord[2]+1))
        '''
        #print('----------------')
        #print(decom_dy)
        for i in range(len(n_u_name_div)):
            for j in range(len(n_u_name_div[i])):
                decom_dy[i][j] = abs(n_u_name_div[i][j].global_d[1][0])
                '''
                if n_u_name_div[i][j].res[0] == 0:
                    decom_dy[i][j] = 1
                else:
                    pass
                '''


        #print('----------------')
        #print(decom_dy)
        #print('----------------')
        print(agent_position)
        #print('----------------')

        # Q1 step1
        for i in range(agent_position[0]):
            for j in range(agent_position[1]):
                decom_dy[i][j+1] += decom_dy[i][j]/2
                decom_dy[i+1][j] += decom_dy[i][j]/2
                #print('Q1 step1')
                #print(decom_dy)
                #print('--------------------------')
        # Q2 step1
        for i in range(agent_position[0]):
            for j in range(len(decom_dy[0])-1,agent_position[1],-1):
                decom_dy[i][j-1] += decom_dy[i][j]/2
                decom_dy[i+1][j] += decom_dy[i][j]/2
                #print('Q2 step1')
                #print(decom_dy)
                #print('--------------------------')
        # Q3 step1
        for i in range(len(decom_dy)-1,agent_position[0],-1):
            for j in range(agent_position[1]):
                decom_dy[i][j+1] += decom_dy[i][j]/2
                decom_dy[i-1][j] += decom_dy[i][j]/2
                #print('Q3 step1')
                #print(decom_dy)
                #print('--------------------------')
        # Q4 step1
        for i in range(len(decom_dy)-1,agent_position[0],-1):
            for j in range(len(decom_dy[0])-1,agent_position[1],-1):
                decom_dy[i][j-1] += decom_dy[i][j]/2
                decom_dy[i-1][j] += decom_dy[i][j]/2
                #print('Q4 step1')
                #print(decom_dy)
                #print('--------------------------')

        # Q1 step2
        '''
        #scheme1
        for i in range(agent_position[0]):
            decom_dy[i+1][agent_position[1]] += decom_dy[i][agent_position[1]] #Add down to the agent
        '''
        #scheme2 stop before the agent
        for i in range(agent_position[0]-1):
            decom_dy[i+1][agent_position[1]] += decom_dy[i][agent_position[1]] #Add down to the agent
        #print('Q1 step2')
        #print(decom_dy)
        #print('--------------------------')
        # Q2 step2
        '''
        #scheme1
        for i in range(len(decom_dy[0])-1,agent_position[1],-1):
            decom_dy[agent_position[0]][i-1] += decom_dy[agent_position[0]][i] #Add left to the agent
        '''
        #scheme2 stop before the agent
        for i in range(len(decom_dy[0])-1,agent_position[1]+1,-1):
            decom_dy[agent_position[0]][i-1] += decom_dy[agent_position[0]][i] #Add left to the agent
        #print('Q2 step2')
        #print(decom_dy)
        #print('--------------------------')
        # Q3 step2
        '''
        #scheme1
        for i in range(len(decom_dy)-1,agent_position[0],-1):
            decom_dy[i-1][agent_position[1]] += decom_dy[i][agent_position[1]] #Add up to the agent
        '''
        #scheme2 stop before the agent
        for i in range(len(decom_dy)-1,agent_position[0]+1,-1):
            decom_dy[i-1][agent_position[1]] += decom_dy[i][agent_position[1]] #Add up to the agent
        #print('Q3 step2')
        #print(decom_dy)
        #print('--------------------------')
        # Q4 step2
        '''
        #scheme1
        for i in range(agent_position[1]-1):
            decom_dy[agent_position[0]][i+1] += decom_dy[agent_position[0]][i] #Add right to the agent
        '''
        #scheme2 stop before the agent
        for i in range(agent_position[1]-1):
            decom_dy[agent_position[0]][i+1] += decom_dy[agent_position[0]][i] #Add right to the agent
        #print('Q4 step2')
        #print(decom_dy)
        #print('--------------------------')


        '''
        decom_q1 = decom_dy[:agent_position[0]+1,:agent_position[1]+1] #top left
        print(decom_q1)
        # Start at [0][0]
        for i in range(len(decom_q1)-1):
            for j in range(len(decom_q1[i])-1):
                decom_q1[i][j+1] += decom_q1[i][j]/2
                decom_q1[i+1][j] += decom_q1[i][j]/2
        for i in range(len(decom_q1)-1):
            decom_q1[i+1][-1] += decom_q1[i][-1]
        for i in range(len(decom_q1[-1])-1):
            decom_q1[-1][i+1] += decom_q1[-1][i]
        print(decom_q1)
        '''
        '''
        print('----------------')
        decom_q2 = decom_dy[:agent_position[0]+1,agent_position[1]:] #top right
        print(decom_q2)
        print('----------------')
        decom_q3 = decom_dy[agent_position[0]:,:agent_position[1]+1] #bottom left
        print(decom_q3)
        print('----------------')
        decom_q4 = decom_dy[agent_position[0]:,agent_position[1]:] #bottom right
        print(decom_q4)
        '''
        # Normalized with sum
        print(decom_dy)
        sum_decom_dy = np.sum(decom_dy)
        decom_dy /= sum_decom_dy
        #print('Normalized')
        #print(decom_dy)
        #print('--------------------------')
        return decom_dy

    # Function to create y-coordinate value from x and z values / x値とz値からy座標値を作成する関数
    def _Y_Val_Range(self,x,z):
        # 研究室
        #self.c6 = np.round(np.random.random(),2)*5/self.c7
        x = x - ((self.num_x-1)/2)
        z = z - ((self.num_z-1)/2)
        return ((self.c1*(x**2) + self.c2*(x*z) + self.c3*(z**2) + self.c4*x + self.c5*z + self.c6 ) * self.c7) #defult
        #return 0.005*x*((x**2)-3*(z**2)) # monkey saddle equation
        #return np.sin(x) + np.sin(z) # trigonometric function
        #return np.sin(0.075*x*z) # trigonometric function

        #return (-0.3*((x**2) + (z**2))) + np.random.random()
        #return random.randint(0,10)/10
        #return 0

    # Function to create coordinate [x,y,z] for every node / すべてのノードの座標[x、y、z]を作成する関数
    '''
    ------------------------------------------
    self.n_u_coord data structure for x=i and z=j / x = iおよびz = jのself.n_u_coordデータ構造
    [
    [[node x1z1],[node x2z1],...,[node xiz1]],
    [[node x1z2],[node x2z2],...,[node xiz2]],
    :
    [[node x1zj],[node x2zj],...,[node xizj]]
    ]
    ------------------------------------------
    '''
    def gennode(self):
        self.n_u_x = []
        self.n_u_z = []
        self.n_u_coord = []

        for i in range(self.num_x):
            self.n_u_x.append(i*self.span)
        for i in range(self.num_z):
            self.n_u_z.append(i*self.span)
        '''
        for i in range(-int(self.num_x/2),int(self.num_x/2)):
            self.n_u_x.append(i*self.span)
        for i in range(-int(self.num_z/2),int(self.num_z/2)):
            self.n_u_z.append(i*self.span)
        '''
        for i in range(len(self.n_u_x)):
            for j in range(len(self.n_u_z)):
                self.n_u_coord.append([self.n_u_x[i],self.span*self._Y_Val_Range(self.n_u_x[i],self.n_u_z[j]),self.n_u_z[j]])

    # Function to export structure data into .txt data / 構造データを.txtデータにエクスポートする関数
    def savetxt(self,name):
        # ------------------------------
        # Write and save output model  / 出力モデルファイルの書き込みと保存
        # ------------------------------
        new_file = open(name, "w+")
        for num1 in range(len(self.model.loads)):
            new_file.write(" {}\r\n".format(self.model.loads[num1]))
        for num1 in range(len(self.model.nodes)):
            new_file.write(" {}\r\n".format(self.model.nodes[num1]))
        for num1 in range(len(self.model.elements)):
            new_file.write(" {},{},{},{},{},{},{},{},{}\r\n".format(
                self.model.elements[num1].name,
                self.model.elements[num1].nodes[0].name,
                self.model.elements[num1].nodes[1].name,
                self.model.elements[num1].em,
                self.model.elements[num1].area,
                self.model.elements[num1].i,
                self.model.elements[num1].sm,
                self.model.elements[num1].j,
                self.model.elements[num1].aor
                ))
        new_file.close()

    # Function to generate frame structure / フレーム構造を生成する機能
    def generate(self):
        '''
        ----------------------------------
        Generate Loads / Loadを生成する
        ----------------------------------
        '''
        l1 = Load()
        l1.set_name(1)
        l1.set_type(1)
        l1.set_size(self.loadx,self.loady,self.loadz,self.loadx,self.loady,self.loadz)
        '''
        ----------------------------------
        Generate Nodes / ノードを生成
        ----------------------------------
        '''
        n = 'n'
        n_u_name=[]
        counter = 1
        for i in range(len(self.n_u_coord)):
            n_u_name.append(n+str(counter))
            n_u_name[-1] = Node()
            n_u_name[-1].set_name(counter)
            n_u_name[-1].set_coord(self.n_u_coord[i][0],self.n_u_coord[i][1],self.n_u_coord[i][2])
            n_u_name[-1].set_res(0,0,0,0,0,0)
            n_u_name[-1].set_hinge(0)
            counter+=1

        #Divide n_u_name into zrow
        for i in range(len(self.n_u_z)):
            self.n_u_name_div.append([])
        for i in range(len(n_u_name)):
            for j in range(len(self.n_u_z)):
                if n_u_name[i].coord[2] == self.n_u_z[j]:
                    self.n_u_name_div[j].append(n_u_name[i])
        '''
        ---------------------------------------------
        Generate Initial shape for reinforcement learning / 強化学習の初期形状を生成する
        ---------------------------------------------
        研究室
        self.n_u_name_div data structure for z=i and x=j / z = iおよびx = jのself.n_u_name_divデータ構造
        [
        [[node z1x1],[node z2x1],...,[node zix1]],
        [[node z1x2],[node z2x2],...,[node zix2]],
        :
        [[node z1xj],[node z2xj],...,[node zixj]]
        ]

        This is template / これはテンプレートです
        To set all surrounding node use this code / すべての周囲のノードを設定するには、このコードを使用します
            ---------------------------------------------
            for i in range(len(self.n_u_name_div)):
                # left
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = value
                # right
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = value
            for i in range(len(self.n_u_name_div[0])):
                # front
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = value
                # back
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = value
            ---------------------------------------------
        To set all corner node use this code / すべてのコーナーノードを設定するには、次のコードを使用します
            ---------------------------------------------
                # corner 1
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = value
                # corner 2
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = value
                # corner 3
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = value
                # corner 4
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1) # boundary condition
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = value
            ---------------------------------------------
        '''
        if self.forgame ==0000:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = i*(self.game_max_y_val/len(self.n_u_name_div))
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = 1-((i+1)*(self.game_max_y_val/len(self.n_u_name_div)))
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/len(self.n_u_name_div[0]))
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = 1-((i+1)*(self.game_max_y_val/len(self.n_u_name_div[0])))

        elif self.forgame == 1:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))

        elif self.forgame == 2:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = 0
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = 0
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))

        elif self.forgame == 3:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                if i < (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][0].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                elif i > (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][0].coord[1] = (len(self.n_u_name_div)-i-1)*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                else:
                    self.n_u_name_div[i][0].coord[1] = self.game_max_y_val
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                if i < (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][-1].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                elif i > (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][-1].coord[1] = (len(self.n_u_name_div)-i-1)*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                else:
                    self.n_u_name_div[i][-1].coord[1] = self.game_max_y_val

            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                if i < (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                elif i > (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[0][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                else:
                    self.n_u_name_div[0][i].coord[1] = self.game_max_y_val
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                if i < (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[-1][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                elif i > (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[-1][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                else:
                    self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val

        elif self.forgame == 4:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = self.game_max_y_val*((i+1)**2) / (len(self.n_u_name_div)**2) #ok

                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = self.game_max_y_val*(((len(self.n_u_name_div)-i)**2) / (len(self.n_u_name_div)**2))
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = self.game_max_y_val*((i+1)**2) / (len(self.n_u_name_div[0])**2)
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val*(((len(self.n_u_name_div[0])-i)**2) / (len(self.n_u_name_div[0])**2))

        elif self.forgame == 5:
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                if i < (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][0].coord[1] = 0
                elif i > (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][0].coord[1] = 0
                else:
                    self.n_u_name_div[i][0].coord[1] = 0
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                if i < (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][-1].coord[1] = 0
                elif i > (len(self.n_u_name_div)-1)/2:
                    self.n_u_name_div[i][-1].coord[1] = 0
                else:
                    self.n_u_name_div[i][-1].coord[1] = 0
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                if i < (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                elif i > (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[0][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                else:
                    self.n_u_name_div[0][i].coord[1] = self.game_max_y_val
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                if i < (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[-1][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                elif i > (len(self.n_u_name_div[0])-1)/2:
                    self.n_u_name_div[-1][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                else:
                    self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val

        elif self.forgame == 999:
            # 4 possibilities of form
            rand_val = random.random()*6
            print('-------------------------------------------------------')
            print('MINIMAL SURFACE GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*(1/self.alpha)))/int(self.game_max_y_val*(1/self.alpha))
            if rand_val>= 5:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = i*(self.game_max_y_val/len(self.n_u_name_div))
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 1-((i+1)*(self.game_max_y_val/len(self.n_u_name_div)))
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/len(self.n_u_name_div[0]))
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 1-((i+1)*(self.game_max_y_val/len(self.n_u_name_div[0])))
            elif rand_val>= 4:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    if i < (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][0].coord[1] = 0
                    elif i > (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][0].coord[1] = 0
                    else:
                        self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    if i < (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][-1].coord[1] = 0
                    elif i > (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][-1].coord[1] = 0
                    else:
                        self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    if i < (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    elif i > (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[0][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    else:
                        self.n_u_name_div[0][i].coord[1] = self.game_max_y_val
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    if i < (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[-1][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    elif i > (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[-1][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    else:
                        self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val
            elif rand_val>= 3:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
            elif rand_val>= 2:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 1-((i-3)**2*(self.game_max_y_val/(len(self.n_u_name_div)+2)))
            elif rand_val>= 1:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    if i < (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][0].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                    elif i > (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][0].coord[1] = (len(self.n_u_name_div)-i-1)*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                    else:
                        self.n_u_name_div[i][0].coord[1] = self.game_max_y_val
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    if i < (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][-1].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                    elif i > (len(self.n_u_name_div)-1)/2:
                        self.n_u_name_div[i][-1].coord[1] = (len(self.n_u_name_div)-i-1)*(self.game_max_y_val/((len(self.n_u_name_div)-1)/2))
                    else:
                        self.n_u_name_div[i][-1].coord[1] = self.game_max_y_val

                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    if i < (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[0][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    elif i > (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[0][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    else:
                        self.n_u_name_div[0][i].coord[1] = self.game_max_y_val
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    if i < (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[-1][i].coord[1] = i*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    elif i > (len(self.n_u_name_div[0])-1)/2:
                        self.n_u_name_div[-1][i].coord[1] = (len(self.n_u_name_div[0])-i-1)*(self.game_max_y_val/((len(self.n_u_name_div[0])-1)/2))
                    else:
                        self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val
            else:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = self.game_max_y_val*((i+1)**2) / (len(self.n_u_name_div)**2) #ok

                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = self.game_max_y_val*(((len(self.n_u_name_div)-i)**2) / (len(self.n_u_name_div)**2))
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = self.game_max_y_val*((i+1)**2) / (len(self.n_u_name_div[0])**2)
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = self.game_max_y_val*(((len(self.n_u_name_div[0])-i)**2) / (len(self.n_u_name_div[0])**2))



        elif self.forgame ==1000:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('DOME --------------------------------------------')
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('ISLER --------------------------------------------')

        elif self.forgame ==1001:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = 0
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = 0
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = 0
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = 0
            print('DOME --------------------------------------------')
        elif self.forgame ==1002:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
            self.n_u_name_div[0][0].set_hinge(0)
            self.n_u_name_div[0][0].coord[1] = 0
            self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
            self.n_u_name_div[0][-1].set_hinge(0)
            self.n_u_name_div[0][-1].coord[1] = 0
            self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
            self.n_u_name_div[-1][0].set_hinge(0)
            self.n_u_name_div[-1][0].coord[1] = 0
            self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
            self.n_u_name_div[-1][-1].set_hinge(0)
            self.n_u_name_div[-1][-1].coord[1] = 0
            print('ISLER --------------------------------------------')

        elif self.forgame ==1003:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 0.50:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 0.25:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')

        elif self.forgame ==1004:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 0.5:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 0.25:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')

        elif self.forgame ==1005:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 0.50:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 0.25:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')

        elif self.forgame ==1006:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                print('2POINT ISLER --------------------------------------------')
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('2POINT ISLER --------------------------------------------')

        elif self.forgame ==1007:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                print('VAULT --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('VAULT --------------------------------------------')
        elif self.forgame ==1008:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                line_pos = []
                dot1_pos = []
                dot2_pos = []
                for i in range(len(self.n_u_name_div)):
                    if i <= len(self.n_u_name_div)/2:
                        line_pos.append(i)
                    else:
                        dot1_pos.append(i)
                        dot2_pos.append(i)
                line_pos = random.choice(line_pos)
                dot1_pos = random.choice(dot1_pos)
                dot2_pos = random.choice(dot2_pos)
                # line
                for i in range(1,len(self.n_u_name_div)-1):
                    self.n_u_name_div[i][line_pos].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][line_pos].set_hinge(0)
                    self.n_u_name_div[i][line_pos].coord[1] = 0
                # dot1
                self.n_u_name_div[1][dot1_pos].set_res(1,1,1,1,1,1)
                self.n_u_name_div[1][dot1_pos].set_hinge(0)
                self.n_u_name_div[1][dot1_pos].coord[1] = 0
                # dot2
                self.n_u_name_div[-2][dot2_pos].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-2][dot2_pos].set_hinge(0)
                self.n_u_name_div[-2][dot2_pos].coord[1] = 0

                print('ONE LINE TWO DOTS ---------------------------------')

            elif rand_val>= 0.5:
                line_pos = []
                dot1_pos = []
                dot2_pos = []
                for i in range(len(self.n_u_name_div)):
                    if i <= len(self.n_u_name_div)/2:
                        dot1_pos.append(i)
                        dot2_pos.append(i)
                    else:
                        line_pos.append(i)

                line_pos = random.choice(line_pos)
                dot1_pos = random.choice(dot1_pos)
                dot2_pos = random.choice(dot2_pos)
                # line
                for i in range(1,len(self.n_u_name_div)-1):
                    self.n_u_name_div[i][line_pos].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][line_pos].set_hinge(0)
                    self.n_u_name_div[i][line_pos].coord[1] = 0
                # dot1
                self.n_u_name_div[1][dot1_pos].set_res(1,1,1,1,1,1)
                self.n_u_name_div[1][dot1_pos].set_hinge(0)
                self.n_u_name_div[1][dot1_pos].coord[1] = 0
                # dot2
                self.n_u_name_div[-2][dot2_pos].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-2][dot2_pos].set_hinge(0)
                self.n_u_name_div[-2][dot2_pos].coord[1] = 0
                print('ONE LINE TWO DOTS ---------------------------------')
            elif rand_val>= 0.25:
                line_pos = []
                dot1_pos = []
                dot2_pos = []
                for i in range(len(self.n_u_name_div[0])):
                    if i <= len(self.n_u_name_div)/2:
                        line_pos.append(i)
                    else:
                        dot1_pos.append(i)
                        dot2_pos.append(i)

                line_pos = random.choice(line_pos)
                dot1_pos = random.choice(dot1_pos)
                dot2_pos = random.choice(dot2_pos)
                # line
                for i in range(1,len(self.n_u_name_div)-1):
                    self.n_u_name_div[line_pos][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[line_pos][i].set_hinge(0)
                    self.n_u_name_div[line_pos][i].coord[1] = 0
                # dot1
                self.n_u_name_div[dot1_pos][1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[dot1_pos][1].set_hinge(0)
                self.n_u_name_div[dot1_pos][1].coord[1] = 0
                # dot2
                self.n_u_name_div[dot2_pos][-2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[dot2_pos][-2].set_hinge(0)
                self.n_u_name_div[dot2_pos][-2].coord[1] = 0
                print('ONE LINE TWO DOTS ---------------------------------')
            else:
                line_pos = []
                dot1_pos = []
                dot2_pos = []
                for i in range(len(self.n_u_name_div[0])):
                    if i <= len(self.n_u_name_div)/2:
                        dot1_pos.append(i)
                        dot2_pos.append(i)
                    else:
                        line_pos.append(i)

                line_pos = random.choice(line_pos)
                dot1_pos = random.choice(dot1_pos)
                dot2_pos = random.choice(dot2_pos)
                # line
                for i in range(1,len(self.n_u_name_div)-1):
                    self.n_u_name_div[line_pos][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[line_pos][i].set_hinge(0)
                    self.n_u_name_div[line_pos][i].coord[1] = 0
                # dot1
                self.n_u_name_div[dot1_pos][1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[dot1_pos][1].set_hinge(0)
                self.n_u_name_div[dot1_pos][1].coord[1] = 0
                # dot2
                self.n_u_name_div[dot2_pos][-2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[dot2_pos][-2].set_hinge(0)
                self.n_u_name_div[dot2_pos][-2].coord[1] = 0
                print('ONE LINE TWO DOTS ---------------------------------')

        elif self.forgame ==1009:
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            # set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定



            # Divide structure into 4 blocks and assign support in each block
            # Block 1
            block1_line_1 = []
            block1_line_2 = []
            block1_dot_pos1 = []
            block1_dot_pos2 = []
            # Block 2
            block2_line_1 = []
            block2_line_2 = []
            block2_dot_pos1 = []
            block2_dot_pos2 = []
            # Block 3
            block3_line_1 = []
            block3_line_2 = []
            block3_dot_pos1 = []
            block3_dot_pos2 = []
            # Block 4
            block4_line_1 = []
            block4_line_2 = []
            block4_dot_pos1 = []
            block4_dot_pos2 = []

            # {
            for i in range(len(self.n_u_name_div)):
                if i <= len(self.n_u_name_div)/2:
                    block1_line_1.append(i)
                    block1_dot_pos1.append(i)

                    block2_line_1.append(i)
                    block2_dot_pos1.append(i)

                if i >= len(self.n_u_name_div)/2:
                    block3_line_1.append(i)
                    block3_dot_pos1.append(i)

                    block4_line_1.append(i)
                    block4_dot_pos1.append(i)


            # [0]{
            for i in range(len(self.n_u_name_div[0])):
                if i <= len(self.n_u_name_div[0])/2:
                    block1_line_2.append(i)
                    block1_dot_pos2.append(i)

                    block3_line_2.append(i)
                    block3_dot_pos2.append(i)

                if i >= len(self.n_u_name_div[0])/2:
                    block2_line_2.append(i)
                    block2_dot_pos2.append(i)

                    block4_line_2.append(i)
                    block4_dot_pos2.append(i)




            # Block 1
            b1_l1 = random.choice(block1_line_1)
            b1_l2 = random.choice(block1_line_2)
            b1_d_p1 = random.choice(block1_dot_pos1)
            b1_d_p2 = random.choice(block1_dot_pos2)
            # Block 2
            b2_l1 = random.choice(block2_line_1)
            b2_l2 = random.choice(block2_line_2)
            b2_d_p1 = random.choice(block2_dot_pos1)
            b2_d_p2 = random.choice(block2_dot_pos2)
            # Block 3
            b3_l1 = random.choice(block3_line_1)
            b3_l2 = random.choice(block3_line_2)
            b3_d_p1 = random.choice(block3_dot_pos1)
            b3_d_p2 = random.choice(block3_dot_pos2)
            # Block 4
            b4_l1 = random.choice(block4_line_1)
            b4_l2 = random.choice(block4_line_2)
            b4_d_p1 = random.choice(block4_dot_pos1)
            b4_d_p2 = random.choice(block4_dot_pos2)





            '''
            # line line dot or none in each block
            # BLOCK 1

            rand_bock1 = random.random()
            if rand_bock1>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i < len(self.n_u_name_div)/2:
                        self.n_u_name_div[i][block1_line_2].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][block1_line_2].set_hinge(0)
                        self.n_u_name_div[i][block1_line_2].coord[1] = 0
            elif rand_bock1>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i < len(self.n_u_name_div[0])/2:
                        self.n_u_name_div[block1_line_1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[block1_line_1][i].set_hinge(0)
                        self.n_u_name_div[block1_line_1][i].coord[1] = 0
            else:
                self.n_u_name_div[block1_dot_pos1][block1_dot_pos2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[block1_dot_pos1][block1_dot_pos2].set_hinge(0)
                self.n_u_name_div[block1_dot_pos1][block1_dot_pos2].coord[1] = 0


            # BLOCK 2
            rand_bock2 = random.random()
            if rand_bock2>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i < len(self.n_u_name_div)/2:
                        self.n_u_name_div[i][block2_line_2].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][block2_line_2].set_hinge(0)
                        self.n_u_name_div[i][block2_line_2].coord[1] = 0
                    else:
                        pass
            elif rand_bock2>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        self.n_u_name_div[block2_line_1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[block2_line_1][i].set_hinge(0)
                        self.n_u_name_div[block2_line_1][i].coord[1] = 0
            else:
                self.n_u_name_div[block2_dot_pos1][block2_dot_pos2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[block2_dot_pos1][block2_dot_pos2].set_hinge(0)
                self.n_u_name_div[block2_dot_pos1][block2_dot_pos2].coord[1] = 0


            # BLOCK 3
            rand_bock3 = random.random()
            if rand_bock3>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i < len(self.n_u_name_div)/2:
                        self.n_u_name_div[block3_line_1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[block3_line_1][i].set_hinge(0)
                        self.n_u_name_div[block3_line_1][i].coord[1] = 0
            elif rand_bock3>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        self.n_u_name_div[i][block3_line_2].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][block3_line_2].set_hinge(0)
                        self.n_u_name_div[i][block3_line_2].coord[1] = 0
            else:
                self.n_u_name_div[block3_dot_pos1][block3_dot_pos2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[block3_dot_pos1][block3_dot_pos2].set_hinge(0)
                self.n_u_name_div[block3_dot_pos1][block3_dot_pos2].coord[1] = 0


            # BLOCK 4
            rand_bock4 = random.random()
            if rand_bock4>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i >= len(self.n_u_name_div)/2:
                        self.n_u_name_div[block4_line_1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[block4_line_1][i].set_hinge(0)
                        self.n_u_name_div[block4_line_1][i].coord[1] = 0
            elif rand_bock4>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        self.n_u_name_div[i][block4_line_2].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][block4_line_2].set_hinge(0)
                        self.n_u_name_div[i][block4_line_2].coord[1] = 0
            else:
                self.n_u_name_div[block4_dot_pos1][block4_dot_pos2].set_res(1,1,1,1,1,1)
                self.n_u_name_div[block4_dot_pos1][block4_dot_pos2].set_hinge(0)
                self.n_u_name_div[block4_dot_pos1][block4_dot_pos2].coord[1] = 0
            '''


            # line line dot or none in each block
            # BLOCK 1

            b1_l1 = random.choice(block1_line_1)
            b1_l2 = random.choice(block1_line_2)
            b1_d_p1 = random.choice(block1_dot_pos1)
            b1_d_p2 = random.choice(block1_dot_pos2)

            rand_bock1 = random.random()
            if rand_bock1>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i <= len(self.n_u_name_div)/2:
                        try:
                            self.n_u_name_div[i][b1_l2].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[i][b1_l2].set_hinge(0)
                            self.n_u_name_div[i][b1_l2].coord[1] = 0
                        except:
                            pass
            elif rand_bock1>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i <= len(self.n_u_name_div[0])/2:
                        try:
                            self.n_u_name_div[b1_l1][i].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[b1_l1][i].set_hinge(0)
                            self.n_u_name_div[b1_l1][i].coord[1] = 0
                        except:
                            pass
            else:
                try:
                    self.n_u_name_div[b1_d_p1][b1_d_p2].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[b1_d_p1][b1_d_p2].set_hinge(0)
                    self.n_u_name_div[b1_d_p1][b1_d_p2].coord[1] = 0
                except:
                    pass


            # BLOCK 2
            rand_bock2 = random.random()
            if rand_bock2>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i <= len(self.n_u_name_div)/2:
                        try:
                            self.n_u_name_div[i][b2_l2].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[i][b2_l2].set_hinge(0)
                            self.n_u_name_div[i][b2_l2].coord[1] = 0
                        except:
                            pass
            elif rand_bock2>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        try:
                            self.n_u_name_div[b2_l1][i].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[b2_l1][i].set_hinge(0)
                            self.n_u_name_div[b2_l1][i].coord[1] = 0
                        except:
                            pass
            else:
                try:
                    self.n_u_name_div[b2_d_p1][b2_d_p2].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[b2_d_p1][b2_d_p2].set_hinge(0)
                    self.n_u_name_div[b2_d_p1][b2_d_p2].coord[1] = 0
                except:
                    pass


            # BLOCK 3
            rand_bock3 = random.random()
            if rand_bock3>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i <= len(self.n_u_name_div)/2:
                        try:
                            self.n_u_name_div[i][b3_l2].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[i][b3_l2].set_hinge(0)
                            self.n_u_name_div[i][b3_l2].coord[1] = 0
                        except:
                            pass
            elif rand_bock3>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        try:
                            self.n_u_name_div[b3_l1][i].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[b3_l1][i].set_hinge(0)
                            self.n_u_name_div[b3_l1][i].coord[1] = 0
                        except:
                            pass
            else:
                try:
                    self.n_u_name_div[b3_d_p1][b3_d_p2].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[b3_d_p1][b3_d_p2].set_hinge(0)
                    self.n_u_name_div[b3_d_p1][b3_d_p2].coord[1] = 0
                except:
                    pass


            # BLOCK 4
            rand_bock4 = random.random()
            if rand_bock4>2/3:
                for i in range(len(self.n_u_name_div)):
                    if i >= len(self.n_u_name_div)/2:
                        try:
                            self.n_u_name_div[i][b4_l2].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[i][b4_l2].set_hinge(0)
                            self.n_u_name_div[i][b4_l2].coord[1] = 0
                        except:
                            pass
            elif rand_bock4>1/3:
                for i in range(len(self.n_u_name_div[0])):
                    if i >= len(self.n_u_name_div[0])/2:
                        try:
                            self.n_u_name_div[b4_l1][i].set_res(1,1,1,1,1,1)
                            self.n_u_name_div[b4_l1][i].set_hinge(0)
                            self.n_u_name_div[b4_l1][i].coord[1] = 0
                        except:
                            pass
            else:
                try:
                    self.n_u_name_div[b4_d_p1][b4_d_p2].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[b4_d_p1][b4_d_p2].set_hinge(0)
                    self.n_u_name_div[b4_d_p1][b4_d_p2].coord[1] = 0
                except:
                    pass

            print('Not because they are easy, but because they are hard---')


        elif self.forgame ==1999:
            # 7 possibilities of form
            rand_val = random.random()*7
            print('-------------------------------------------------------')
            print('GENERATE GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = 0
            if rand_val>= 6:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                # Set Z axis to 0 / Z軸を0に設定
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('DOME --------------------------------------------')
            elif rand_val>= 5:
                # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
                # Set Z axis to 0 / Z軸を0に設定
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('ISLER --------------------------------------------')
            elif rand_val>= 4:
                # 3side dome
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                elif rand_val2>= 2:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                elif rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 3:
                #3POINT ISLER
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                elif rand_val2>= 2:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                elif rand_val2>= 1:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                else:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 2:
                #2SIDES DOME
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                elif rand_val2>= 2:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                elif rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 1:
                #2POINT ISLER
                rand_val2 = random.random()*2
                if rand_val2>= 1:
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    print('2POINT ISLER --------------------------------------------')
                else:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('2POINT ISLER --------------------------------------------')
            else:
                #VAULT
                rand_val2 = random.random()*2
                if rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    print('VAULT --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('VAULT --------------------------------------------')

        elif self.forgame ==2000:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('DOME --------------------------------------------')
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('ISLER --------------------------------------------')

        elif self.forgame ==2001:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][0].coord[1] = 0
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
                self.n_u_name_div[i][-1].coord[1] = 0
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[0][i].coord[1] = 0
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
                self.n_u_name_div[-1][i].coord[1] = 0
            print('DOME --------------------------------------------')
        elif self.forgame ==2002:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
            self.n_u_name_div[0][0].set_hinge(0)
            self.n_u_name_div[0][0].coord[1] = 0
            self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
            self.n_u_name_div[0][-1].set_hinge(0)
            self.n_u_name_div[0][-1].coord[1] = 0
            self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
            self.n_u_name_div[-1][0].set_hinge(0)
            self.n_u_name_div[-1][0].coord[1] = 0
            self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
            self.n_u_name_div[-1][-1].set_hinge(0)
            self.n_u_name_div[-1][-1].coord[1] = 0
            print('ISLER --------------------------------------------')

        elif self.forgame ==2003:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 0.50:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 0.25:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('3SIDES DOME --------------------------------------------')

        elif self.forgame ==2004:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 0.5:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 0.25:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                print('3POINT ISLER --------------------------------------------')

        elif self.forgame ==2005:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.75:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 0.50:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 0.25:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                print('2SIDES DOME --------------------------------------------')

        elif self.forgame ==2006:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                print('2POINT ISLER --------------------------------------------')
            else:
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('2POINT ISLER --------------------------------------------')

        elif self.forgame ==2007:
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Random initial node height within maximum y value / 最大y値内のランダムな初期ノードの高さ
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            # 3-side Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            rand_val = random.random()
            if rand_val>= 0.5:
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                print('VAULT --------------------------------------------')
            else:
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('VAULT --------------------------------------------')
        elif self.forgame ==2999:
            # 7 possibilities of form
            rand_val = random.random()*7
            print('-------------------------------------------------------')
            print('OPTIMIZATION GAME') #研究室
            print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            for i in range(len(self.n_u_name_div)):
                for j in range(len(self.n_u_name_div[i])):
                    self.n_u_name_div[i][j].coord[1] = random.randint(0,int(self.game_max_y_val*10))/int(self.game_max_y_val*10)
            if rand_val>= 6:
                # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
                # Set Z axis to 0 / Z軸を0に設定
                for i in range(len(self.n_u_name_div)):
                    self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][0].set_hinge(0)
                    self.n_u_name_div[i][0].coord[1] = 0
                    self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[i][-1].set_hinge(0)
                    self.n_u_name_div[i][-1].coord[1] = 0
                for i in range(len(self.n_u_name_div[0])):
                    self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][i].set_hinge(0)
                    self.n_u_name_div[0][i].coord[1] = 0
                    self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][i].set_hinge(0)
                    self.n_u_name_div[-1][i].coord[1] = 0
                print('DOME --------------------------------------------')
            elif rand_val>= 5:
                # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
                # Set Z axis to 0 / Z軸を0に設定
                self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][0].set_hinge(0)
                self.n_u_name_div[0][0].coord[1] = 0
                self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][-1].set_hinge(0)
                self.n_u_name_div[0][-1].coord[1] = 0
                self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][0].set_hinge(0)
                self.n_u_name_div[-1][0].coord[1] = 0
                self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][-1].set_hinge(0)
                self.n_u_name_div[-1][-1].coord[1] = 0
                print('ISLER --------------------------------------------')
            elif rand_val>= 4:
                # 3side dome
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                elif rand_val2>= 2:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                elif rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('3SIDES DOME --------------------------------------------')
            elif rand_val>= 3:
                #3POINT ISLER
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                elif rand_val2>= 2:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                elif rand_val2>= 1:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
                else:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    print('3POINT ISLER --------------------------------------------')
            elif rand_val>= 2:
                #2SIDES DOME
                rand_val2 = random.random()*4
                if rand_val2>= 3:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                elif rand_val2>= 2:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                elif rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                    print('2SIDES DOME --------------------------------------------')
            elif rand_val>= 1:
                #2POINT ISLER
                rand_val2 = random.random()*2
                if rand_val2>= 1:
                    self.n_u_name_div[0][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][-1].set_hinge(0)
                    self.n_u_name_div[0][-1].coord[1] = 0
                    self.n_u_name_div[-1][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][0].set_hinge(0)
                    self.n_u_name_div[-1][0].coord[1] = 0
                    print('2POINT ISLER --------------------------------------------')
                else:
                    self.n_u_name_div[0][0].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[0][0].set_hinge(0)
                    self.n_u_name_div[0][0].coord[1] = 0
                    self.n_u_name_div[-1][-1].set_res(1,1,1,1,1,1)
                    self.n_u_name_div[-1][-1].set_hinge(0)
                    self.n_u_name_div[-1][-1].coord[1] = 0
                    print('2POINT ISLER --------------------------------------------')
            else:
                #VAULT
                rand_val2 = random.random()*2
                if rand_val2>= 1:
                    for i in range(len(self.n_u_name_div)):
                        self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][0].set_hinge(0)
                        self.n_u_name_div[i][0].coord[1] = 0
                        self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[i][-1].set_hinge(0)
                        self.n_u_name_div[i][-1].coord[1] = 0
                    print('VAULT --------------------------------------------')
                else:
                    for i in range(len(self.n_u_name_div[0])):
                        self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[0][i].set_hinge(0)
                        self.n_u_name_div[0][i].coord[1] = 0
                        self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                        self.n_u_name_div[-1][i].set_hinge(0)
                        self.n_u_name_div[-1][i].coord[1] = 0
                    print('VAULT --------------------------------------------')

        else:
            # 研究室
            # Use 'Function to create y-coordinate value from x and z values' / 「関数を使用してx値とz値からy座標値を作成する」
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            for i in range(len(self.n_u_name_div)):
                self.n_u_name_div[i][0].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][0].set_hinge(0)
                self.n_u_name_div[i][-1].set_res(1,1,1,1,1,1)
                self.n_u_name_div[i][-1].set_hinge(0)
            for i in range(len(self.n_u_name_div[0])):
                self.n_u_name_div[0][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[0][i].set_hinge(0)
                self.n_u_name_div[-1][i].set_res(1,1,1,1,1,1)
                self.n_u_name_div[-1][i].set_hinge(0)
        #Set node name /ノード名を設定
        node_pool =[]
        counter = 1
        for i in range(len(n_u_name)):
            node_pool.append(n_u_name[i])
            node_pool[-1].set_name(counter)
            node_pool[-1].set_load(l1)
            counter +=1
            '''
            if node_pool[-1].res[0] != 1:
                node_pool[-1].set_load(l1)
            else:
                pass
            '''

        '''
        ----------------------------------
        Generate elements / 要素を生成する
        ----------------------------------
        '''
        e = 'e'
        E_type1_name =[]
        counter = 1
        for num in range(len(self.n_u_name_div)):
            for i in range((len(self.n_u_name_div[0])-1)):
                E_type1_name.append(e+str(counter))
                E_type1_name[-1] = Element()
                E_type1_name[-1].set_name(counter)
                E_type1_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num][i+1])
                E_type1_name[-1].set_em(self.Young)
                E_type1_name[-1].set_area(self.area)
                E_type1_name[-1].set_i(self.ival,self.ival,self.ival)
                E_type1_name[-1].set_j(self.jval)
                E_type1_name[-1].set_sm(self.shearmodulus)
                E_type1_name[-1].set_aor(0)
                counter+=1

        E_type2_name =[]
        counter = len(E_type1_name)+1
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])):
                E_type2_name.append(e+str(counter))
                E_type2_name[-1] = Element()
                E_type2_name[-1].set_name(counter)
                E_type2_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i])
                E_type2_name[-1].set_em(self.Young)
                E_type2_name[-1].set_area(self.area)
                E_type2_name[-1].set_i(self.ival,self.ival,self.ival)
                E_type2_name[-1].set_j(self.jval)
                E_type2_name[-1].set_sm(self.shearmodulus)
                E_type2_name[-1].set_aor(0)
                counter+=1

        E_type3_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+1
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])-1):
                E_type3_name.append(e+str(counter))
                E_type3_name[-1] = Element()
                E_type3_name[-1].set_name(counter)
                E_type3_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i+1])
                E_type3_name[-1].set_em(self.Young)
                E_type3_name[-1].set_area(self.area)
                E_type3_name[-1].set_i(self.ival,self.ival,self.ival)
                E_type3_name[-1].set_j(self.jval)
                E_type3_name[-1].set_sm(self.shearmodulus)
                E_type3_name[-1].set_aor(0)
                #E_type3_name[-1].set_aor(45)
                counter+=1
        E_type4_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+1
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])-1):
                E_type4_name.append(e+str(counter))
                E_type4_name[-1] = Element()
                E_type4_name[-1].set_name(counter)
                E_type4_name[-1].set_nodes(self.n_u_name_div[num][i+1],self.n_u_name_div[num+1][i])
                E_type4_name[-1].set_em(self.Young)
                E_type4_name[-1].set_area(self.area)
                E_type4_name[-1].set_i(self.ival,self.ival,self.ival)
                E_type4_name[-1].set_j(self.jval)
                E_type4_name[-1].set_sm(self.shearmodulus)
                E_type4_name[-1].set_aor(0)
                counter+=1




        '''
        ----------------------------------
        Generate Structural Model / 構造モデルを生成
        ----------------------------------
        '''
        self.model = Model()
        self.model.add_load(l1)
        for i in range(len(n_u_name)):
            self.model.add_node(n_u_name[i])
        for i in range(len(E_type1_name)):
            self.model.add_element(E_type1_name[i])
        for i in range(len(E_type2_name)):
            self.model.add_element(E_type2_name[i])
        if self.brace!=None:
            # diagonal elements /対角要素
            for i in range(len(E_type3_name)):
                self.model.add_element(E_type3_name[i])
            for i in range(len(E_type4_name)):
                self.model.add_element(E_type4_name[i])

        #----------------------------------
        # Set selfweight on each element
        '''
        l1 = Load()
        l1.set_name(1)
        l1.set_type(1)
        l1.set_size(self.loadx,self.loady,self.loadz,self.loadx,self.loady,self.loadz)
        '''
        if self.use_selfweight == True:
            w = 'w'
            w_type1_name =[]
            counter = 1
            for num in range(len(self.model.elements)):
                self.model.elements[num].gen_length()
                # Case end node higher than start node
                if self.model.elements[num].nodes[1].coord[1] > self.model.elements[num].nodes[0].coord[1]:
                    # weight in each direction
                    weight_y = self.weight
                    weight_x = self.weight
                    # perpendicular uniform distributed load in y-axis
                    w_type1_name.append(w+str(counter))
                    w_type1_name[-1] = Load()
                    w_type1_name[-1].set_name(counter+1)
                    w_type1_name[-1].set_type(2)# 2 = perpendicular uniform distributed load
                    w_type1_name[-1].set_size(0,-weight_y,0,0,-weight_y,0)
                    self.model.elements[num].set_load(w_type1_name[-1],0,0) #[load,distance_from_startnode,distance_from_endnode]self on member
                    counter+=1

                    # axial uniform load in x-axis
                    w_type1_name.append(w+str(counter))
                    w_type1_name[-1] = Load()
                    w_type1_name[-1].set_name(counter+1)
                    w_type1_name[-1].set_type(2)# 2 = perpendicular uniform distributed load
                    w_type1_name[-1].set_size(-weight_x,0,0,-weight_x,0,0)
                    self.model.elements[num].set_load(w_type1_name[-1],0,0)
                    counter+=1

                # Case end node lower than start node
                elif self.model.elements[num].nodes[1].coord[1] > self.model.elements[num].nodes[0].coord[1]:
                    # weight in each direction
                    weight_y = self.weight
                    weight_x = self.weight
                    # perpendicular uniform distributed load in y-axis
                    w_type1_name.append(w+str(counter))
                    w_type1_name[-1] = Load()
                    w_type1_name[-1].set_name(counter+1)
                    w_type1_name[-1].set_type(2)# 2 = perpendicular uniform distributed load
                    w_type1_name[-1].set_size(0,-weight_y,0,0,-weight_y,0)
                    self.model.elements[num].set_load(w_type1_name[-1],0,0)
                    counter+=1

                    # axial uniform load in x-axis
                    w_type1_name.append(w+str(counter))
                    w_type1_name[-1] = Load()
                    w_type1_name[-1].set_name(counter+1)
                    w_type1_name[-1].set_type(2)# 2 = perpendicular uniform distributed load
                    w_type1_name[-1].set_size(weight_x,0,0,weight_x,0,0)
                    self.model.elements[num].set_load(w_type1_name[-1],0,0)
                    counter+=1

                # Case end node equal start node
                elif self.model.elements[num].nodes[1].coord[1] == self.model.elements[num].nodes[0].coord[1]:
                    # weight in each direction
                    weight_y = self.weight
                    weight_x = self.weight
                    # perpendicular uniform distributed load in y-axis
                    w_type1_name.append(w+str(counter))
                    w_type1_name[-1] = Load()
                    w_type1_name[-1].set_name(counter+1)
                    w_type1_name[-1].set_type(2)# 2 = perpendicular uniform distributed load
                    w_type1_name[-1].set_size(0,-weight_y,0,0,-weight_y,0)
                    self.model.elements[num].set_load(w_type1_name[-1],0,0)
                    counter+=1

            for i in range(len(w_type1_name)):
                self.model.add_load(w_type1_name[i])

        #----------------------------------

        self.model.gen_all()
        return self.model

    def gen_surface1(self):
        '''
        ----------------------------------
        Generate surface area with method 1 /
        ----------------------------------
        '''
        self.surface_1 = 0
        for i in range(len(self.n_u_name_div)-1):
            for j in range(len(self.n_u_name_div[i])-1):
                '''
                # A NODE:
                x = self.n_u_name_div[i][j].coord[0]
                y = self.n_u_name_div[i][j].coord[1]
                z = self.n_u_name_div[i][j].coord[2]
                # B NODE:
                x = self.n_u_name_div[i+1][j].coord[0]
                y = self.n_u_name_div[i+1][j].coord[1]
                z = self.n_u_name_div[i+1][j].coord[2]
                # C NODE:
                x = self.n_u_name_div[i+1][j+1].coord[0]
                y = self.n_u_name_div[i+1][j+1].coord[1]
                z = self.n_u_name_div[i+1][j+1].coord[2]
                # D NODE:
                x = self.n_u_name_div[i][j+1].coord[0]
                y = self.n_u_name_div[i][j+1].coord[1]
                z = self.n_u_name_div[i][j+1].coord[2]
                '''
                # VALUE OF VECTOR AB,AC,AD
                Xab = ((self.n_u_name_div[i+1][j].coord[0]-self.n_u_name_div[i][j].coord[0])**2)**0.5
                Yab = ((self.n_u_name_div[i+1][j].coord[1]-self.n_u_name_div[i][j].coord[1])**2)**0.5
                Zab = ((self.n_u_name_div[i+1][j].coord[2]-self.n_u_name_div[i][j].coord[2])**2)**0.5
                Xac = ((self.n_u_name_div[i+1][j+1].coord[0]-self.n_u_name_div[i][j].coord[0])**2)**0.5
                Yac = ((self.n_u_name_div[i+1][j+1].coord[1]-self.n_u_name_div[i][j].coord[1])**2)**0.5
                Zac = ((self.n_u_name_div[i+1][j+1].coord[2]-self.n_u_name_div[i][j].coord[2])**2)**0.5
                Xad = ((self.n_u_name_div[i][j+1].coord[0]-self.n_u_name_div[i][j].coord[0])**2)**0.5
                Yad = ((self.n_u_name_div[i][j+1].coord[1]-self.n_u_name_div[i][j].coord[1])**2)**0.5
                Zad = ((self.n_u_name_div[i][j+1].coord[2]-self.n_u_name_div[i][j].coord[2])**2)**0.5
                # Area of a triangle ABC
                Sabc = 0.5*((((Yab*Zac)-(Zab*Yac))**2+((Zab*Xac)-(Xab*Zac))**2+((Xab*Yac)-(Yab*Xac))**2)**0.5)
                # Area of a triangle ACD
                Sabd = 0.5*((((Yac*Zad)-(Zac*Yad))**2+((Zac*Xad)-(Xac*Zad))**2+((Xac*Yad)-(Yac*Xad))**2)**0.5)
                # VALUE OF VECTOR DA,DB,DC
                Xda = ((self.n_u_name_div[i][j].coord[0]-self.n_u_name_div[i][j+1].coord[0])**2)**0.5
                Yda = ((self.n_u_name_div[i][j].coord[1]-self.n_u_name_div[i][j+1].coord[1])**2)**0.5
                Zda = ((self.n_u_name_div[i][j].coord[2]-self.n_u_name_div[i][j+1].coord[2])**2)**0.5
                Xdb = ((self.n_u_name_div[i+1][j].coord[0]-self.n_u_name_div[i][j+1].coord[0])**2)**0.5
                Ydb = ((self.n_u_name_div[i+1][j].coord[1]-self.n_u_name_div[i][j+1].coord[1])**2)**0.5
                Zdb = ((self.n_u_name_div[i+1][j].coord[2]-self.n_u_name_div[i][j+1].coord[2])**2)**0.5
                Xdc = ((self.n_u_name_div[i+1][j+1].coord[0]-self.n_u_name_div[i][j+1].coord[0])**2)**0.5
                Ydc = ((self.n_u_name_div[i+1][j+1].coord[1]-self.n_u_name_div[i][j+1].coord[1])**2)**0.5
                Zdc = ((self.n_u_name_div[i+1][j+1].coord[2]-self.n_u_name_div[i][j+1].coord[2])**2)**0.5
                # Area of a triangle DAB
                Sdab = 0.5*((((Yda*Zdb)-(Zda*Ydb))**2+((Zda*Xdb)-(Xda*Zdb))**2+((Xda*Ydb)-(Yda*Xdb))**2)**0.5)
                # Area of a triangle DBC
                Sdbc = 0.5*((((Ydb*Zdc)-(Zdb*Ydc))**2+((Zdb*Xdc)-(Xdb*Zdc))**2+((Xdb*Ydc)-(Ydb*Xdc))**2)**0.5)


                #self.surface_1 += (Sabc+Sdab)/2
                #self.surface_1 += (Sabd+Sdbc)/2
                self.surface_1 += (Sabc+Sabd+Sdab+Sdbc)/2

        self.surface_1 = [self.surface_1]
        '''
        ----------------------------------
        Generate surface area with method 2: shpae function /
        ----------------------------------
        '''
        '''
        ip_alpha = [-1,1]
        ip_beta = [-1,1]
        for i in range(len(self.n_u_name_div)-1):
            for j in range(len(self.n_u_name_div[i])-1):
                # alpha, beta = +1
                dN1da_p1 = 0.25*(-1+1)
                dN2da_p1 = 0.25*(1-1)
                dN3da_p1 = 0.25*(1+1)
                dN4da_p1 = 0.25*(-1-1)

                dN1db_p1 = 0.25*(-1+1)
                dN2db_p1 = 0.25*(-1-1)
                dN3db_p1 = 0.25*(1+1)
                dN4db_p1 = 0.25*(1-1)
                # alpha, beta = -1
                dN1da_m1 = 0.25*(-1+(-1))
                dN2da_m1 = 0.25*(1-(-1))
                dN3da_m1 = 0.25*(1+(-1))
                dN4da_m1 = 0.25*(-1-(-1))

                dN1db_m1 = 0.25*(-1+(-1))
                dN2db_m1 = 0.25*(-1-(-1))
                dN3db_m1 = 0.25*(1+(-1))
                dN4db_m1 = 0.25*(1-(-1))
        '''

    # Function to render in matplotlib / ＃matplotlibでレンダリングする関数
    def render(self,name='',iteration=0,initenergy=0,recentenergy=0):
        #========================
        #  Elevation
        #========================
        fig = plt.figure(figsize=plt.figaspect(0.3))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.75, wspace=0, hspace=None)
        ax1 = fig.add_subplot(1,3,1,projection='3d')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')

        ax1.view_init(azim=90, elev=0)
        #ax1.view_init(azim=0, elev=90)
        #Model Name
        Analysis_type0 = name
        Structure_type0 = 'Structure type : Space Frame'
        Model_code0 = 'Size : ' + str(len(self.model.nodes))+' nodes ' + str(len(self.model.elements))+' members'
        Node_and_Member = 'Elevation-1'
        # NODES
        Xcoord1 = []  #Horizontal
        Ycoord1 = []  #Vertical
        Zcoord1 = []  #Horizontal
        for i in range(len(self.model.nodes)):
            Xcoord1.append(self.model.nodes[i].coord[0])
            Ycoord1.append(self.model.nodes[i].coord[2]) #In Input and Analysis Z = Horizotal
            Zcoord1.append(self.model.nodes[i].coord[1]) #In Input and Analysis Y = Vertical
        Nodes1 = {'X':Xcoord1, 'Y':Ycoord1, 'Z':Zcoord1}
        nodeplot1 = pd.DataFrame(Nodes1, columns = ['X','Y','Z'])
        fignodeplot1 = ax1.scatter(xs=nodeplot1['X'],ys=nodeplot1['Y'],zs=nodeplot1['Z'],color ='red',s=1)
        # ELEMENTS
        for i in range(len(self.model.elements)):
            xstart = self.model.elements[i].nodes[0].coord[0]
            xend = self.model.elements[i].nodes[1].coord[0]
            ystart = self.model.elements[i].nodes[0].coord[2] #In Input and Analysis Z = Horizotal
            yend = self.model.elements[i].nodes[1].coord[2] #In Input and Analysis Z = Horizotal
            zstart = self.model.elements[i].nodes[0].coord[1] #In Input and Analysis Y = Vertical
            zend = self.model.elements[i].nodes[1].coord[1] #In Input and Analysis Y = Vertical
            Xcoord = [xstart,xend]
            Zcoord = [zstart,zend]
            Ycoord = [ystart,yend]
            Elements = {'X':Xcoord, 'Y':Ycoord, 'Z':Zcoord}
            elementplot = pd.DataFrame(Elements, columns = ['X','Y','Z'])
            figelementplot = ax1.plot(xs=elementplot['X'],ys=elementplot['Y'],zs=elementplot['Z'],color ='red')
        # SUPPORT
        #CASE 1 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_Zres1=[]
        Ycoord_Zres1=[]
        Zcoord_Zres1=[]
        #CASE 2 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_ZresXres1=[]
        Ycoord_ZresXres1=[]
        Zcoord_ZresXres1=[]
        #CASE 3 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresYres1=[]
        Ycoord_ZresYres1=[]
        Zcoord_ZresYres1=[]
        #CASE 4 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresXresYres1=[]
        Ycoord_ZresXresYres1=[]
        Zcoord_ZresXresYres1=[]
        for i in range(len(self.model.nodes)):
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_Zres1.append(self.model.nodes[i].coord[0])
                Ycoord_Zres1.append(self.model.nodes[i].coord[2])
                Zcoord_Zres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_ZresXres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresYres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresYres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresYres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresXresYres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXresYres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXresYres1.append(self.model.nodes[i].coord[1])
            else:
                pass
        if Xcoord_Zres1 != 0:
            Zres1 = {'X':Xcoord_Zres1, 'Y':Ycoord_Zres1, 'Z':Zcoord_Zres1}
            Zresplot1 = pd.DataFrame(Zres1, columns = ['X','Y','Z'])
            figZresplot1 = ax1.scatter(xs=Zresplot1['X'],ys=Zresplot1['Y'],zs=Zresplot1['Z'],color ='black',marker="^",s=100)
        else:
            pass
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([nodeplot1['X'].max()-nodeplot1['X'].min(), nodeplot1['Y'].max()-nodeplot1['Y'].min(), nodeplot1['Z'].max()-nodeplot1['Z'].min()]).max()
        Xb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.01*(nodeplot1['X'].max()+nodeplot1['X'].min())
        Yb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.01*(nodeplot1['Y'].max()+nodeplot1['Y'].min())
        Zb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.01*(nodeplot1['Z'].max()+nodeplot1['Z'].min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax1.plot([xb], [yb], [zb], 'w')
        plt.title(Analysis_type0+'\n'+Structure_type0+'\n'+Model_code0+'\n'+'\n'+Node_and_Member,loc='left')
        plt.grid()
        #plt.axis('equal')
        #========================
        #  Elevation2
        #========================
        axe2 = fig.add_subplot(1,3,2,projection='3d')
        axe2.set_xlabel('X axis')
        axe2.set_ylabel('Y axis')
        axe2.set_zlabel('Z axis')

        axe2.view_init(azim=180, elev=0)
        #ax1.view_init(azim=0, elev=90)
        #Model Name
        Analysis_type0 = ''
        Structure_type0 = ''
        Model_code0 = ''
        Node_and_Member = 'Elevation-2'
        # NODES
        Xcoord1 = []  #Horizontal
        Ycoord1 = []  #Vertical
        Zcoord1 = []  #Horizontal
        for i in range(len(self.model.nodes)):
            Xcoord1.append(self.model.nodes[i].coord[0])
            Ycoord1.append(self.model.nodes[i].coord[2]) #In Input and Analysis Z = Horizotal
            Zcoord1.append(self.model.nodes[i].coord[1]) #In Input and Analysis Y = Vertical
        Nodes1 = {'X':Xcoord1, 'Y':Ycoord1, 'Z':Zcoord1}
        nodeplot1 = pd.DataFrame(Nodes1, columns = ['X','Y','Z'])
        fignodeplot1 = axe2.scatter(xs=nodeplot1['X'],ys=nodeplot1['Y'],zs=nodeplot1['Z'],color ='red',s=1)
        # ELEMENTS
        for i in range(len(self.model.elements)):
            xstart = self.model.elements[i].nodes[0].coord[0]
            xend = self.model.elements[i].nodes[1].coord[0]
            ystart = self.model.elements[i].nodes[0].coord[2] #In Input and Analysis Z = Horizotal
            yend = self.model.elements[i].nodes[1].coord[2] #In Input and Analysis Z = Horizotal
            zstart = self.model.elements[i].nodes[0].coord[1] #In Input and Analysis Y = Vertical
            zend = self.model.elements[i].nodes[1].coord[1] #In Input and Analysis Y = Vertical
            Xcoord = [xstart,xend]
            Zcoord = [zstart,zend]
            Ycoord = [ystart,yend]
            Elements = {'X':Xcoord, 'Y':Ycoord, 'Z':Zcoord}
            elementplot = pd.DataFrame(Elements, columns = ['X','Y','Z'])
            figelementplot = axe2.plot(xs=elementplot['X'],ys=elementplot['Y'],zs=elementplot['Z'],color ='red')
        # SUPPORT
        #CASE 1 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_Zres1=[]
        Ycoord_Zres1=[]
        Zcoord_Zres1=[]
        #CASE 2 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_ZresXres1=[]
        Ycoord_ZresXres1=[]
        Zcoord_ZresXres1=[]
        #CASE 3 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresYres1=[]
        Ycoord_ZresYres1=[]
        Zcoord_ZresYres1=[]
        #CASE 4 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresXresYres1=[]
        Ycoord_ZresXresYres1=[]
        Zcoord_ZresXresYres1=[]
        for i in range(len(self.model.nodes)):
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_Zres1.append(self.model.nodes[i].coord[0])
                Ycoord_Zres1.append(self.model.nodes[i].coord[2])
                Zcoord_Zres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_ZresXres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresYres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresYres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresYres1.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresXresYres1.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXresYres1.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXresYres1.append(self.model.nodes[i].coord[1])
            else:
                pass
        if Xcoord_Zres1 != 0:
            Zres1 = {'X':Xcoord_Zres1, 'Y':Ycoord_Zres1, 'Z':Zcoord_Zres1}
            Zresplot1 = pd.DataFrame(Zres1, columns = ['X','Y','Z'])
            figZresplot1 = axe2.scatter(xs=Zresplot1['X'],ys=Zresplot1['Y'],zs=Zresplot1['Z'],color ='black',marker="^",s=100)
        else:
            pass
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([nodeplot1['X'].max()-nodeplot1['X'].min(), nodeplot1['Y'].max()-nodeplot1['Y'].min(), nodeplot1['Z'].max()-nodeplot1['Z'].min()]).max()
        Xb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.01*(nodeplot1['X'].max()+nodeplot1['X'].min())
        Yb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.01*(nodeplot1['Y'].max()+nodeplot1['Y'].min())
        Zb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.01*(nodeplot1['Z'].max()+nodeplot1['Z'].min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           axe2.plot([xb], [yb], [zb], 'w')
        plt.title(Analysis_type0+'\n'+Structure_type0+'\n'+Model_code0+'\n'+'\n'+Node_and_Member,loc='left')
        plt.grid()
        #plt.axis('equal')
        #========================
        #  INITIAL + DEFORMATION
        #========================
        ax = fig.add_subplot(1,3,3,projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        #ax.view_init(azim=90, elev=0)
        #Model Name
        Analysis_type2 = 'Iteration: '+str(iteration)
        Structure_type2 = 'Base Strain Energy     : ' + str(round(initenergy,9))
        Model_code2 =     'Current Strain Energy : ' + str(round(recentenergy,9))
        # NODES
        Xcoord = []  #Horizontal
        Ycoord = []  #Vertical
        Zcoord = []  #Horizontal
        for i in range(len(self.model.nodes)):
            Xcoord.append(self.model.nodes[i].coord[0])
            Ycoord.append(self.model.nodes[i].coord[2]) #In Input and Analysis Z = Horizotal
            Zcoord.append(self.model.nodes[i].coord[1]) #In Input and Analysis Y = Vertical
        Nodes = {'X':Xcoord, 'Y':Ycoord, 'Z':Zcoord}
        nodeplot = pd.DataFrame(Nodes, columns = ['X','Y','Z'])
        fignodeplot = ax.scatter(xs=nodeplot['X'],ys=nodeplot['Y'],zs=nodeplot['Z'],color ='black',s=1)
        # ELEMENTS
        for i in range(len(self.model.elements)):
            xstart = self.model.elements[i].nodes[0].coord[0]
            xend = self.model.elements[i].nodes[1].coord[0]
            ystart = self.model.elements[i].nodes[0].coord[2] #In Input and Analysis Z = Horizotal
            yend = self.model.elements[i].nodes[1].coord[2] #In Input and Analysis Z = Horizotal
            zstart = self.model.elements[i].nodes[0].coord[1] #In Input and Analysis Y = Vertical
            zend = self.model.elements[i].nodes[1].coord[1] #In Input and Analysis Y = Vertical
            Xcoord = [xstart,xend]
            Zcoord = [zstart,zend]
            Ycoord = [ystart,yend]
            Elements = {'X':Xcoord, 'Y':Ycoord, 'Z':Zcoord}
            elementplot = pd.DataFrame(Elements, columns = ['X','Y','Z'])
            figelementplot = ax.plot(xs=elementplot['X'],ys=elementplot['Y'],zs=elementplot['Z'],color ='black')
        # SUPPORT
        #CASE 1 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_Zres=[]
        Ycoord_Zres=[]
        Zcoord_Zres=[]
        #CASE 2 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0)
        Xcoord_ZresXres=[]
        Ycoord_ZresXres=[]
        Zcoord_ZresXres=[]
        #CASE 3 (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresYres=[]
        Ycoord_ZresYres=[]
        Zcoord_ZresYres=[]
        #CASE 4 (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1)
        Xcoord_ZresXresYres=[]
        Ycoord_ZresXresYres=[]
        Zcoord_ZresXresYres=[]
        for i in range(len(self.model.nodes)):
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_Zres.append(self.model.nodes[i].coord[0])
                Ycoord_Zres.append(self.model.nodes[i].coord[2])
                Zcoord_Zres.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 0):
                Xcoord_ZresXres.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXres.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXres.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresYres.append(self.model.nodes[i].coord[0])
                Ycoord_ZresYres.append(self.model.nodes[i].coord[2])
                Zcoord_ZresYres.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1) and (self.model.nodes[i].res[2] == 1):
                Xcoord_ZresXresYres.append(self.model.nodes[i].coord[0])
                Ycoord_ZresXresYres.append(self.model.nodes[i].coord[2])
                Zcoord_ZresXresYres.append(self.model.nodes[i].coord[1])
            else:
                pass
        if Xcoord_Zres != 0:
            Zres = {'X':Xcoord_Zres, 'Y':Ycoord_Zres, 'Z':Zcoord_Zres}
            Zresplot = pd.DataFrame(Zres, columns = ['X','Y','Z'])
            figZresplot = ax.scatter(xs=Zresplot['X'],ys=Zresplot['Y'],zs=Zresplot['Z'],color ='black',marker="^",s=100)
        else:
            pass
        # DEFROMATION
        # NODES
        NewXcoord = []
        NewYcoord = []
        NewZcoord = []
        for i in range(len(self.model.nodes)):
            NewXcoord.append(self.model.nodes[i].coord[0])
            NewYcoord.append(self.model.nodes[i].coord[2])
            NewZcoord.append(self.model.nodes[i].coord[1])


        for i in range(len(self.model.d)):
            for j in range(len(self.model.tnsc)):
                if self.model.tnsc[j][0] == i + 1:
                    NewXcoord[j] += self.model.d[i][0]
                if self.model.tnsc[j][2] == i + 1:
                    NewYcoord[j] += self.model.d[i][0]
                if self.model.tnsc[j][1] == i + 1:
                    NewZcoord[j] += self.model.d[i][0]



        NewNodes = {'X':NewXcoord, 'Y':NewYcoord, 'Z':NewZcoord}
        Newnodeplot = pd.DataFrame(NewNodes, columns = ['X', 'Y', 'Z'])
        figNewnodeplot = ax.scatter(xs=Newnodeplot['X'],ys=Newnodeplot['Y'],zs=Newnodeplot['Z'],color ='red',alpha=0.5,s=1)
        # ELEMENTS
        for i in range(len(self.model.elements)):
            Newxstart = self.model.elements[i].nodes[0].coord[0]
            Newxend = self.model.elements[i].nodes[1].coord[0]
            Newystart = self.model.elements[i].nodes[0].coord[2]
            Newyend = self.model.elements[i].nodes[1].coord[2]
            Newzstart = self.model.elements[i].nodes[0].coord[1]
            Newzend = self.model.elements[i].nodes[1].coord[1]

            for k in range(len(self.model.d)):
                if self.model.tnsc[self.model.elements[i].nodes[0].name-1][0] == k + 1:
                    Newxstart += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[1].name-1][0] == k + 1:
                    Newxend += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[0].name-1][2] == k + 1:
                    Newystart += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[1].name-1][2] == k + 1:
                    Newyend += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[0].name-1][1] == k + 1:
                    Newzstart += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[1].name-1][1] == k + 1:
                    Newzend += self.model.d[k][0]
                else:
                    pass
            NewXcoord = [Newxstart,Newxend]
            NewZcoord = [Newzstart,Newzend]
            NewYcoord = [Newystart,Newyend]
            NewElements = {'X':NewXcoord, 'Y':NewYcoord, 'Z':NewZcoord}
            Newelementplot = pd.DataFrame(NewElements, columns = ['X','Y','Z'])
            figNewelementplot = ax.plot(xs=Newelementplot['X'],ys=Newelementplot['Y'],zs=Newelementplot['Z'],color ='red',alpha=0.5)
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([nodeplot['X'].max()-nodeplot['X'].min(), nodeplot['Y'].max()-nodeplot['Y'].min(), nodeplot['Z'].max()-nodeplot['Z'].min()]).max()
        Xb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.01*(nodeplot['X'].max()+nodeplot['X'].min())
        Yb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.01*(nodeplot['Y'].max()+nodeplot['Y'].min())
        Zb = 0.01*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.01*(nodeplot['Z'].max()+nodeplot['Z'].min())
        # Comment or uncomment following both lines to test the fake bounding box:

        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')


        plt.title(Analysis_type2+'\n'+Structure_type2+'\n'+Model_code2+'\n'+'\n'+'Perspective',loc='left')
        plt.grid()
        #plt.axis('equal')
        plt.show()

        #plt.savefig(name)
        plt.close("all")



