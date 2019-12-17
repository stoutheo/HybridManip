# Hybrid-Manipulation (HybridManip)

The repository is currently under development.


## 1. Create an environment for HOLM primitives repository
1. Create the environment for the repository.<br/> 
&nbsp;&nbsp;&nbsp;For example, call it py3holm and install python 3 (*hence the name*).<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; with conda : ```conda create -n py3holm python=3.7```<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; with pip : ```python3 -m venv py3holm```<br/><br/>


## 2. Install all the packages for HOLM primitives repository
1. Activate the environment.<br/> 
&nbsp;&nbsp;&nbsp; with conda : ```source activate py3holm```<br/> 
&nbsp;&nbsp;&nbsp; with pip : ```source py3holm/bin/activate```
2. Install all required packages. We have done a *holm.req* file for you to use:<br/> 
&nbsp;&nbsp;&nbsp; with conda : ```conda install --file holm.req```<br/> 
&nbsp;&nbsp;&nbsp; with pip : ```python3 -m pip install -r holm.req```<br/><br/>

## Packages Dependencies 

* Numpy : with conda : ```conda install numpy```,  with pip : ```$ python3 -m pip install -r numpy``` 
* Matplotlib : with conda : ```conda install matplotlib```,  with pip : ```python3 -m pip install -r matplotlib``` 
* yaml (pyyaml): with conda : ```conda install pyyaml```,  with pip : ```python3 -m pip install -r pyyaml``` 
* CasADi : with conda : ```conda install casadi```,  with pip : ```python3 -m pip install -r casadi``` 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for more details please see: https://web.casadi.org  <br/><br/>


## Run examples: 

* activate the created environment (py3holm)
* go inside the folder "HybridManip"

To run the examples:
* ```python3 -m py_pack.examples.*name of example file*```

*e.g.* for one end-effector for Cnt2Sw HOLM primitive (first phase is contact and followed by swing phase)
* ```python3 -m py_pack.examples.Cnt2Sw_HOLM```


*e.g.* for two end-effector agent for Cnt2Cnt and Cnt2Cnt HOLM primitive (first phase is contact and followed by contact phase for both end-effectors)
* ```python3 -m py_pack.examples.Bimanual_Cnt2Cnt_Cnt2Cnt``` <br/><br/>


#### Change parameters, like objects, friction coefficient, *etc*:

* go to file parameters.yml, found in py_pack/config 


#### Change parameters of the problem, like task, initial conditions, variable bounds, *etc*:

* go to file each example file, found in py_pack/example 



