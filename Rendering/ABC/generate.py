import bpy
import numpy as np
import os
import bpy
import sys
import pdb
import string
import json
import argparse

from mathutils import Matrix
from math import degrees
from math import radians 

fpath = bpy.data.filepath
dir_path = '/'.join(fpath.split('/')[:-2])
util_path = os.path.join(dir_path, 'utils')
sys.path.append(util_path)

import render_utils

from render_utils import apply_rot

bpy.context.user_preferences.addons['cycles'].preferences['compute_device_type'] = 1

# only global; loading render params
data_gen_params = render_utils.load_json(
                        os.path.join(dir_path, 
                                     'ABC', 
                                     'data_generation_parameters.json')
                        )

prefs = bpy.context.user_preferences.addons['cycles'].preferences
def generate():
    
    # loading parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_fname', type=str, help='number of views to be rendered')
    parser.add_argument('--gpu', type=int, help='number of views to be rendered')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    obj_fname = args.obj_fname
    
    for idx in range(len(prefs.devices)):
        if idx == args.gpu:
            prefs.devices[idx].use = True
        else:
            prefs.devices[idx].use = False
    
    # loading settings
    render_params = data_gen_params['render_parameters']
    light_params = data_gen_params['light_parameters']
    output_path = data_gen_params['paths']['output_path'] 
    abc_path = data_gen_params['paths']['ABC_path']
    
    file_loc = os.path.join(abc_path, obj_fname)
    
    bpy.ops.import_scene.obj(filepath=file_loc)
        
    #joining object mesh
    objects = bpy.data.objects
    target_object_parts = []
    for obj in objects:
        if obj.name == 'Camera':
            continue
        if obj.name == 'Plane':
            continue
        target_object_parts.append(obj)

    bpy.context.scene.objects.active = None

    for obj in target_object_parts:
        obj.select = True

    bpy.context.scene.objects.active = target_object_parts[0]
    bpy.ops.object.join()

    obj = [obj for obj in bpy.data.objects if (obj.name != 'Camera' and obj.name != 'Plane')][0]
    obj.name = 'object'

    metadata_save_path = os.path.join(output_path, str(obj_fname), 'metadata.txt')

    #setting variables
    scn = bpy.context.scene
    tree = bpy.context.scene.node_tree
    
    #set origin to bounding box
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    # clear normals in .obj file
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # recompute normals
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()

    #force smooth shading
    bpy.ops.object.shade_smooth()

    obj.location = (0,0,0)

    #make sure object fits in cube with side 2
    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.4/np.max(np.abs(vertices))

    render_utils.add_principled_material(bpy.data)
    obj.data.materials.append(bpy.data.materials['principled'])

    #apply render settings
    render_utils.apply_settings(scn, render_params)
    
    #make lamp
    lamp = render_utils.make_area_lamp(data_gen_params['light_parameters']['area_light_location'], 
                                       (0,0,0), 
                                       size_x = data_gen_params['light_parameters']['area_size_x'], 
                                       size_y = data_gen_params['light_parameters']['area_size_x'], 
                                       strength = data_gen_params['light_parameters']['area_strength_default'], 
                                       temp = data_gen_params['light_parameters']['area_temp_default'])
    
    if data_gen_params['gen_params']['jitter_lights']:

        point_locations = data_gen_params['light_parameters']['point_light_locations']
        loc_idx = np.random.choice(len(point_locations))
        location = point_locations[loc_idx] + np.random.uniform(-0.25, 0.25, size=3)
        print(location)
        point_lamp = render_utils.make_point_lamp(location,
                                                  strength = 0,
                                                  temp = 0,
                                                  jitter_location = False,
                                                  shadow_size = 0)

        lamps = [x for x in objects if x.type == 'LAMP']


    if data_gen_params['gen_params']['lambertian']:
        render_utils.remove_materials()
        #add material
        mat = render_utils.add_lambertian_material(bpy.data)
        obj.data.materials.append(mat)
    
    if data_gen_params['gen_params']['jitter_reflectance']:
        materials = bpy.data.materials
        render_utils.tap_materials(materials, data_gen_params['material_parameters'])

    #apply output paths to node_tree
    render_utils.set_output_paths(tree, output_path, obj_fname, 
                                  data_gen_params['gen_params']['outputs'])
    
    # camera and light are constant, object is rotating

    #set object position to canonical pose
    obj.rotation_euler = (0, np.radians(-180), 0)
    
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)

    #make output dir

    if not os.path.exists(os.path.join(output_path, obj_fname)):
        os.makedirs(os.path.join(output_path, obj_fname))


    if data_gen_params['gen_params']['3DOF_vc'] == True:
        

        _3DOF_vc_metadata_save_path = os.path.join(output_path, 
                                                  str(obj_fname), 
                                                  '_3DOF_vc_metadata.txt')

        np.savetxt(_3DOF_vc_metadata_save_path, 
                   np.array([[init_rot], [init_el]]).T, 
                   fmt='%-10.2f', 
                   newline='\n', 
                   header='init azim    init elev')

        init_rot = np.random.uniform(0,360)
        init_el = np.random.uniform(-90, 90)

        render_utils.apply_rot(obj, 'Y', init_rot)
        render_utils.apply_rot(obj, 'X', init_el)

    bpy.context.scene.objects.active = obj
    obj.select = True
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    
    rotations = np.random.uniform(low=data_gen_params['gen_params']['azim_range'][0], 
                                  high=data_gen_params['gen_params']['azim_range'][1], 
                                  size=data_gen_params['gen_params']['n_points'])
    
    elevations = np.random.uniform(low=data_gen_params['gen_params']['elev_range'][0], 
                                   high=data_gen_params['gen_params']['elev_range'][1], 
                                   size=data_gen_params['gen_params']['n_points'])
    
    np.savetxt(metadata_save_path, 
               np.array([rotations, elevations]).T, 
               fmt='%-10.2f', 
               newline='\n', 
               header='azim      elev')

    if data_gen_params['gen_params']['debug']:
        rotations = np.linspace(0, 360, num=25)
        elevations = np.ones(25)*45
    
    for i, (rot, el) in enumerate(zip(rotations, elevations)):
            render_utils.apply_rot(obj, 'Y', rot)
            render_utils.apply_rot(obj, 'X', el)
            
            if data_gen_params['gen_params']['jitter_lights']:
                render_utils.jitter_lights(lamps, 
                                          data_gen_params['light_parameters'])

            bpy.context.scene.update()
            scn.frame_current = i
            bpy.ops.render.render()
            
            render_utils.reset_rot(obj)
            bpy.context.scene.update()
    
    if 'normal' not in data_gen_params['gen_params']['outputs']:
        return

    # rendering normals 
    scn.render.layers['RenderLayer'].cycles.use_denoising = False

    mat_path = os.path.join(dir_path, 'blend_files/materials.blend')
    materials = render_utils.load_materials(mat_path)
    render_utils.assign_material(obj, 'normals_material')
    
    scn.display_settings.display_device = 'None'
    scn.sequencer_colorspace_settings.name = 'Linear'

    for key in obj.cycles_visibility.keys():
        obj.cycles_visibility[key] = 0

    render_utils.adjust_links(tree)

    vertices = []
    for i, (rot, el) in enumerate(zip(rotations, elevations)):
            render_utils.apply_rot(obj, 'Y', rot)
            render_utils.apply_rot(obj, 'X', el)

            bpy.context.scene.update()
            scn.frame_current = i
            bpy.ops.render.render()

            render_utils.reset_rot(obj)
            bpy.context.scene.update()

    
   
if __name__ == '__main__':
    generate()



