![shapenet-examples](ShapeNet/shapenet_examples.png)

# Rendering ShapeNet Objects

## Editing `data_generation_parameters.json`

Once the ShapeNetCore.v2 zip file files is downloaded, the first step is to make the appropriate edits to `data_generation_paramters.json` which is the rendering config file.

- `light_parameters`, `render_parameters`, and `material parameters` have reasonable default values that result in images as featured above. Editing them and doing small test renders is the best way to see the effect they have on the output. `render_parameters` are set up to ensure fast GPU rendering.
- The values under `camera` are `focal_length`, `sensor_size` and `distance_units` which is camera distance. All values are in Blender units.
- `paths`
    - `blender_path`- path to the Blender executable that's been downloaded based on the directions above
    - `shapenet_path` - path to the unzipped ShapeNet dataset.
    - `output_path` - path where the renders will be saved. 
- `gen_params`
    - `elev_range` - elevation range for sampling object views; default is `[-50, 50]`
    - `azim_range` - azimuth range for sampling object views; default is `[0, 360]`
    - `n_points`- number of images generated in object poses sampled from the range defined by the previous two values
    - `debug` - in this mode the object is rendered at a fixed elevation with uniform azimuth increments
    - `jitter_lights` - whether to vary the lights or not within the defined parameters in `light_parameters`
    - `lambertian` - an option to render all objects with a white lambertian shader.
    - `3DOF_vc` - randomly rotate the object once before sampling in `azim_range` and `elev_range`. This allows for greater variability in object poses accross the dataset. 
    - `outputs` - list of `["image", "albedo", "normal", "depth_absolute", "depth_01", "segmentation"]` which defines the types of outputs that should be generated. `depth_absolute` indicates absolute depth images and `depth_01` indicates relative depth images where `min=0` and `max=1`.

### Running Data Generation
```
usage: wrapper.py [-h] [-start START] [-end END] -out_file OUT_FILE [-v] -gpu
                  GPU

Range of Objects

optional arguments:
  -h, --help          show this help message and exit
  -start START        start point in data list
  -end END            end point in data list
  -out_file OUT_FILE  file to output progress to
  -v                  verbose: print or supress blender output
  -gpu GPU            gpu index to use
  ```
 
`wrapper.py` builds a list of all of objects, and the `-start` and `-end` arguments can be used to define which range of objects to render. This is useful when rendering using multiple runs on multiple GPUs.
`run_data_generation.py` can be used to run a big rendering job over multiple GPUs without having to manually start multiple instances of `wrapper.py`.

### Output Structure

For each object, a directory will be created under `output_path` in `data_generation_parameters.json` and a subdirectory for each value in `outputs` of `gen_params`. There is also a `metadata.txt` file that's output that contains the pose information of the object for each image rendered that can be read using `np.loadtxt`.

```
# azim      elev
57.12      -47.35
174.08     18.83
15.66      48.86
38.00      30.44
312.67     -11.42
...
```
