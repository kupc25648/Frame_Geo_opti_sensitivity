# Frame_Geo_opti_sensitivity
Frame structure's geometry optimization using sensitivity analysis

This is the frame structure's geometry optimization program  using sensitivity analysis. The program optimize the strain energy of the frame structure by adjusting nodal heights.

The repository includes 4 files

FEM_frame.py : 3D Frame structural analysis file
frame_GEN.py : Frame structure generate file
master.py : Main file


After the optimization process, the master.py will create 2 outputs
1. A test file contain structural data of 
  1.1 Nodal Load
  1.2 Node
  1.3 Element
2. A matplotlib render of the structural data
<br>
<img src="src/Sensitivity Optimization Solution.png">
<br/>
