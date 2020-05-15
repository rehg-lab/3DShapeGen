### Environment Setup
Create environment using [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
```bash
conda env create -f environment.yml
```
Compile OccNet extension modules in `mesh_gen_utils`
```bash
python setup.py build_ext --inplace
```
To generate ground truths and perform testing, change the path in `isosurface/LIB_PATH` to your miniconda/anaconda libraries, for example
```bash
export LD_LIBRARY_PATH="<path_to_anaconda>/lib:<path_to_anaconda>/envs/sdf_net/lib:./isosurface:$LD_LIBRARY_PATH" 
source isosurface/LIB_PATH
```
Note that the following ground truths generation, training and testing procedures apply to experiments on both ShapeNet and ABC.
### Generating SDF Ground truths and Pointclouds
```
usage: create_sdf.py [-h] [--mesh_dir MESH_DIR]
                     [--norm_mesh_dir NORM_MESH_DIR] [--sdf_dir SDF_DIR]
                     [--json_path JSON_PATH] [--mode MODE]
                     [--categories CATEGORIES] [--num_samples NUM_SAMPLES]
                     [--bandwidth BANDWIDTH] [--res RES]
                     [--expand_rate EXPAND_RATE] [--iso_val ISO_VAL]
                     [--max_verts MAX_VERTS] [--ish5 ISH5]
                     [--normalize NORMALIZE] [--skip_all_exist SKIP_ALL_EXIST]
                     [--ptcl PTCL] [--ptcl_save_dir PTCL_SAVE_DIR]
                     [--ptcl_size PTCL_SIZE] [--num_split NUM_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --mesh_dir MESH_DIR   Orginal mesh directory
  --norm_mesh_dir NORM_MESH_DIR
                        Directory to save normalized mesh
  --sdf_dir SDF_DIR     Directory to save sdf
  --json_path JSON_PATH
                        Path to json file
  --mode MODE           Generating mode (train, val, test). If None all 3 are
                        generated
  --categories CATEGORIES
                        Short-handed categories to generate ground-truth
  --num_samples NUM_SAMPLES
                        Number of sdf sampled
  --bandwidth BANDWIDTH
                        Bandwidth of sampling
  --res RES             Resolution of grid to sample sdf
  --expand_rate EXPAND_RATE
                        Max value of x,y,z
  --iso_val ISO_VAL     Iso surface value
  --max_verts MAX_VERTS
                        Maximum number of vertices
  --ish5 ISH5           Whether to save in h5 format
  --normalize NORMALIZE
                        Whether to normalize gt mesh
  --skip_all_exist SKIP_ALL_EXIST
                        Whether to skip existing ground-truth
  --ptcl PTCL           Whether to generate pointcloud
  --ptcl_save_dir PTCL_SAVE_DIR
                        Where to save pointclouds
  --ptcl_size PTCL_SIZE
                        Size of pointcloud
  --num_split NUM_SPLIT
                        Number of threads to use
```
Example command to generate ground-truth sdf with pointclouds
```bash
python gt_gen/create_sdf.py --mesh_dir=../../ShapeNetCore.v2/ --norm_mesh_dir=../../gen_mesh --sdf_dir=../../gen_sdf --json_path=../../data.json --mode=test --ptcl_save_dir=../../gen_ptcl
```
To generate ground-truth and pointclouds separately
```bash
python gt_gen/create_sdf.py --mesh_dir=../../ShapeNetCore.v2/ --norm_mesh_dir=../../gen_mesh --sdf_dir=../../gen_sdf --json_path=../../data.json --mode=test --ptcl=False
python gt_gen/generate_ptcld.py --mesh_dir=../../gen_mesh --json_path=../../data.json --save_dir=../../gen_ptcl
```
### Training SDFNet
After changing the parameters in `config.py` run the following to train the model from scratch
```bash
python train.py
```
### Pre-trained SDFNet
Pre-trained SDFNet models can be downloaded [here](link-to-download)
### Testing SDFNet
```bash
python eval.py
python read_eval_output.py
vim results/<eval_task_name>.txt
```
This project uses code based on parts of the following repositories

1. [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks)
2. [DISN](https://github.com/Xharlie/DISN)
3. Our F-Score implementation is based on [What Do Single-view 3D Reconstruction Networks Learn?](https://github.com/lmb-freiburg/what3d)