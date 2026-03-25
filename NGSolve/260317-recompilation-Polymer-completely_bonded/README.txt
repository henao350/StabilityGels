### Load results for making the measurements (a), (b), (c), (d), (e) ###
load-results.ipynb
(the thickness should be specified in the Main section of the .ipynb code (Cell 2), as well as the indexes of delta to be loaded)  


### 
### Solve Gels equations  - generates gridfunctions saved in .gfu files ###
###
# already computed #
python3 RUN-Solve_gel3d.py 90 3.00 23.5 2 0 8         computed in Lenovo, 5 Sep
python3 RUN-Solve_gel3d.py 90 3.00 23.5 2 0 7         computed in Lenovo Sep 7 09:16
python3 RUN-Solve_gel3d.py 90 3.00 23.5 2 0 6         computed in Rancagua, 8 Sep, aprox 16 h, out_d3_delta0.7.txt 
python3 RUN-Solve_gel3d.py 90 3.00 23.5 2 0 5	      computed in Rancagua, 8 Sep, 16h23m, out-d3_00-delta0_6.txt
python3 RUN-Solve_gel3d.py 90 3.00 23.5 2 0 4	      computed in Rancagua, 9 Sep, 16h27m, out-d3_00-delta0_5.txt
python3 RUN-Solve_gel3d.py 90 3.00 15.0 2 0 3 	      computed in Rancagua, 10 Sep, 17h29m, out-d3_00-delta0_4.txt
python3 RUN-Solve_gel3d.py 90 3.00 15.0 2 0 2         computed in HPC-UOH, 12 Sep, 92h14m, slurm-288323_3.out
python3 RUN-Solve_gel3d.py 90 3.00 15.0 2 0 1         computed in Rancagua, 13 Sep, 18h18m, out-d3_00-delta0_2.txt
python3 RUN-Solve_gel3d.py 90 3.00 15.0 2 0 0         computed in Rancagua, 13 Sep, 18h28m, out-d3_00-delta0_1.txt
python3 RUN-Solve_gel3d-completely_bonded.py 90 3.00 23.5 2 0 9 	computed in Rancagua, 16 Sep, 16h34m, out-d3_00-delta1_0.txt

python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 8         computed in Rancagua, 8 Sep, aprox 16 h, out_d=1_62_delta=0.9.txt 
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 7         computed in Rancagua 8 Sep, out-d1_62-delta0.8.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 6         computed in Lenovo 8 Sep, aprox 12h, out-d1_62-delta0.7.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 5 	      computed in Lenovo, 8 Sep, 12h10m, out-d1_62-delta0_6.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 4	      computed in Lenovo, 9 Sep, 17h23m, out-d1_62-delta0_5.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 3 	      computed in Lenovo, 10 Sep, 18h27m, out-d1_62-delta0_4.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 2	      computed in Rancagua, 11 Sep, 11h09m, out-d1_62-delta0_3.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 1	      computed in Lenovo, 11 Sep, 13h52, out-d1_62-delta0_2.txt
python3 RUN-Solve_gel3d.py 90 1.62 15.0 2 0 0 	      computed in Rancagua, 11 Sep, 11h38m, out-d1_62-delta0_1.txt
python3 RUN-Solve_gel3d-completely_bonded.py 90 1.62 15.0 2 0 9 	computed in Rancagua, 16 Sep, 12h36m, out-d1_62-delta1_0.txt 


### MESHES ###
Fine meshes (coarse_flag=0):
https://drive.google.com/drive/folders/1ojxTsewNif5th2vdalbpbrDrSlKB60TS?usp=share_link

Moderately coarse mesh, d=3.00 mm:
https://drive.google.com/drive/folders/1g-GgNgva4OZYkAx97e8gJ29aQ-KCIsyh?usp=share_link

Coarse meshes (coarse_flag=1) d = 1.62 mm:
https://drive.google.com/drive/folders/1rZlWN2oyNhg6eCJO4zV8j5y33sG6QBgL?usp=share_link

Coarse meshes (coarse_flag=1) d = 3.00 mm:
https://drive.google.com/drive/folders/1kc00ELomlDBgcpLsCnEqxgwqipEIBqlI?usp=share_link


###
The completely bonded simulation was computed with mesh9.vol.gz (from generate_mesh.py) and imposing a zero Dirichlet condition for u_1 and u_3 also in the debonded part,
i.e. 
    self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded|debonded")
instead of 
    self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded", dirichlety="debonded")
This is 
    RUN-Solve_gel3d-completely_bonded.py
The others were solved eith
    RUN-Solve_gel3d.py
The value delta=1.0 was added to meshes/deltas


### Commands for doing it in the cluster at O'Higgins
screen
sbatch --array=1-8 myarrayjob.sh


