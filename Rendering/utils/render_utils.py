import bpy
import numpy as np
import os
import bpy
import sys
import pdb
import string
import json
import colorsys

from mathutils import Matrix
from math import degrees
from math import radians 

def apply_rot(obj, axis, angle):
    """                                                               
        inputs:                                                       
            obj   - bpy.data.objects object to rotate globally        
            axis  - axis along which to rotate - 'X', 'Y', or 'Z'     
            angle - angle in global coordinates along axis in degrees 
    """                                                               
       
    rot_mat = Matrix.Rotation(radians(angle), 4, axis)
                                                      
    o_loc, o_rot, o_scl = obj.matrix_world.decompose()
    o_loc_mat = Matrix.Translation(o_loc)             
    o_rot_mat = o_rot.to_matrix().to_4x4()
    o_scl_mat = (Matrix.Scale(o_scl[0],4,(1,0,0))
               * Matrix.Scale(o_scl[1],4,(0,1,0))
               * Matrix.Scale(o_scl[2],4,(0,0,1)))
                                                 
    # assemble the new matrix                                         
    obj.matrix_world = o_loc_mat * rot_mat * o_rot_mat * o_scl_mat    

def make_area_lamp(location, rotation, size_x = 0, size_y = 0, strength = 10, temp = 5000):
    """
    inputs:
        location --- (x,y,z) where to place the lamp
        rotation --- (x,y,z) euler for the lamp rotation
        size_x   --- size in one direction
        size_y   --- size in the other direction
        strength --- brightness (test-render some images for good value)
        temp     --- temperature in kelvin
    """
    #make sure nothing is active or selected
    bpy.context.scene.objects.active = None
    bpy.ops.object.select_all(action='DESELECT')
    #add lamp
    bpy.ops.object.lamp_add(type='AREA', 
                            view_align=False, 
                            location=location, 
                            rotation = rotation, 
                            layers=(True, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False))

    #since lamp was just added it will be active object, set its shape and size
    lamp = bpy.data.lamps[bpy.context.active_object.name]
    lamp.shape  = 'RECTANGLE'
    lamp.size   = size_x 
    lamp.size_y = size_y
    
    #define short variable
    nodes = lamp.node_tree.nodes
    
    #remove the default nodes that show up
    for node in nodes:
        nodes.remove(node)    
    #add new nodes to control strength (emission) and blackbody (color temperature) and output
    node_blackbody = nodes.new(type= 'ShaderNodeBlackbody')
    node_emission  = nodes.new(type = 'ShaderNodeEmission')
    node_output    = nodes.new(type = 'ShaderNodeOutputLamp')

    #placing the nodes within the node diagram
    node_output.location[1] = 400
    node_emission.location[1] = 200

    # connect blackbody to emission color, connect emission to output
    lamp.node_tree.links.new(node_blackbody.outputs[0],node_emission.inputs[0])
    lamp.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])

    # set node values to specified values
    node_emission.inputs[1].default_value = strength
    node_blackbody.inputs[0].default_value = temp

    return lamp

def delete_lamps():
	# deselect all
	bpy.ops.object.select_all(action='DESELECT')

	# selection and deletion
	for obj in bpy.data.objects:

		if obj.type == 'LAMP':
			obj.select = True
			bpy.ops.object.delete()

def make_point_lamp(location, strength = 100, temp = 5000, jitter_location = False, shadow_size=0.0):

    if jitter_location == True:
            location = location + np.random.uniform(-1,1,3)

    bpy.ops.object.lamp_add(type='POINT', 
                            view_align=False, 
                            location=location, 
                            layers=(True, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False, 
                                    False, False, False, False))

    lamp = bpy.data.lamps[bpy.context.active_object.name]

    lamp.shadow_soft_size = shadow_size

    nodes = lamp.node_tree.nodes

    for node in nodes:
            nodes.remove(node)

    node_blackbody = nodes.new(type= 'ShaderNodeBlackbody')
    node_emission  = nodes.new(type = 'ShaderNodeEmission')
    node_output    = nodes.new(type = 'ShaderNodeOutputLamp')

    node_output.location[1] = 400
    node_emission.location[1] = 200

    lamp.node_tree.links.new(node_blackbody.outputs[0],node_emission.inputs[0])
    lamp.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])

    node_emission.inputs[1].default_value = strength
    node_blackbody.inputs[0].default_value = temp

    return lamp

def tap_materials(materials, material_parameters):
    
    specular_range = material_parameters['specular_range']
    rougness_range = material_parameters['rougness_range']
    specular_val = np.random.uniform(*specular_range)
    roughness_val = np.random.uniform(*rougness_range)

    for material in materials:
        print(material.name)

        assert isinstance(material.node_tree, bpy.types.ShaderNodeTree)

        tree = material.node_tree
        nodes = tree.nodes

        target_node = None
        mixcolor_node = None
        normal_node = None
        shadermix_node = None

        for node in nodes:

            if node.label == 'Diff BSDF':
                target_node = node
            
            if node.label == 'Mix Color/Diffuse':
                mixcolor_node = node
            
            if node.label == 'Normal/Map':
                normal_node = node
            
            if node.label == 'Shader Mix Alpha':
                shadermix_node = node

        assert target_node is not None
        assert mixcolor_node is not None
        assert normal_node is not None
        assert shadermix_node is not None

        nodes.remove(target_node)

        PrincipledBSDF = material.node_tree.nodes.new(type = 'ShaderNodeBsdfPrincipled')

        normal_input = next((x for x in PrincipledBSDF.inputs if x.name == 'Normal'), None)
        color_input = next((x for x in PrincipledBSDF.inputs if x.name == 'Base Color'), None)
        specular_input = next((x for x in PrincipledBSDF.inputs if x.name == 'Specular'), None)
        roughness_input = next((x for x in PrincipledBSDF.inputs if x.name == 'Roughness'), None)
        bsdf_output = PrincipledBSDF.outputs[0]

        bsdf_input = shadermix_node.inputs[-1]
        normal_output = normal_node.outputs[0]
        color_output = mixcolor_node.outputs[0]

        tree.links.new(bsdf_output, bsdf_input)
        tree.links.new(normal_output, normal_input)
        tree.links.new(color_output, color_input)
        
        specular_input.default_value = specular_val
        roughness_input.default_value = roughness_val

def jitter_lights(lamps, light_parameters):
    
    area_strength_range = light_parameters['area_strength_range']
    point_strength_range = light_parameters['point_strength_range']
    temp_range = light_parameters['light_temperature_range']
    point_locations = light_parameters['point_light_locations']
    point_size_range = light_parameters['point_size_range']
    
    lamp_type = np.random.choice(light_parameters['light_types'])
    
    #lamps = [x for x in lamps if x.data.type == lamp_type]

    for lamp in lamps:
        lamp.hide_render = True

        if lamp_type == 'POINT':
            strength = np.random.randint(*point_strength_range)
            temp = np.random.randint(*temp_range)
            loc_idx = np.random.choice(len(point_locations))
            location = point_locations[loc_idx] + np.random.uniform(-0.5, 0.5, size=3) 
            point_size = np.random.uniform(*point_size_range)

            jitter_point_light(lamp,
                               strength = strength,
                               temp = temp,
                               location = location,
                               point_size = point_size)

            lamp.hide_render = False

        elif lamp_type == 'AREA':
            strength = np.random.randint(*point_strength_range)
            temp = np.random.randint(*temp_range)

            jitter_area_light(lamp, strength, temp)

            lamp.hide_render = False
        
        else:
            raise Exception("lamp type can be \'POINT\' or \'AREA\'")


def jitter_area_light(lamp, strength, temp):
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = strength
    lamp.data.node_tree.nodes['Blackbody'].inputs[0].default_value = temp

def jitter_point_light(lamp, strength=None, temp=None, location=None, point_size=None):
    
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = strength
    lamp.data.node_tree.nodes['Blackbody'].inputs[0].default_value = temp

    lamp.data.shadow_soft_size = point_size
    lamp.location = location

def reset_rot(obj):
    obj.rotation_euler = (0,0,0)

def apply_settings(scn, render_parameters):
    
    scn.render.resolution_x = render_parameters['resolution']
    scn.render.resolution_y = render_parameters['resolution']
    scn.render.resolution_percentage = render_parameters['resolution_percentage']
    scn.render.layers['RenderLayer'].samples = render_parameters['render_samples']
    scn.cycles.debug_use_spatial_splits = render_parameters['use_spatial_splits']
    scn.cycles.max_bounces = render_parameters['max_bounces']
    scn.cycles.min_bounces = render_parameters['min_bounces']
    scn.cycles.transparent_max_bounces = render_parameters['transparent_max_bounces']
    scn.cycles.transparent_min_bounces = render_parameters['transparent_min_bounces']
    scn.cycles.glossy_bounces = render_parameters['glossy_bounces']
    scn.cycles.transmission_bounces = render_parameters['transmission_bounces']
    scn.render.use_persistent_data = render_parameters['use_persistent_data']
    scn.render.tile_x = render_parameters['render_tile_x']
    scn.render.tile_y = render_parameters['render_tile_y']
    scn.cycles.caustics_refractive = render_parameters['use_caustics_refractive']
    scn.cycles.caustics_reflective = render_parameters['use_caustics_reflective']
    scn.cycles.device = render_parameters['rendering_device']
    scn.render.image_settings.color_mode = render_parameters['color_mode']
    scn.render.layers['RenderLayer'].cycles.use_denoising = render_parameters['use_denoising']
    scn.render.layers['RenderLayer'].cycles.denoising_radius = render_parameters['denoising_radius']
    scn.cycles.film_transparent = render_parameters['use_film_transparent']

def set_output_paths_ABC(tree, output_path, obj_fname, outputs):

    segmentation_output = tree.nodes['Segmentation']
    image_output = tree.nodes['Image_Output']
    depth_output = tree.nodes['Depth_Output']
    normal_output = tree.nodes['Normal_Output']
    albedo_output = tree.nodes['Albedo_Output']
    exr_output = tree.nodes['OpenEXR_Output']

    if 'image' not in outputs:
        l = image_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        image_output.base_path = os.path.join(output_path, obj_fname, 'image_output')

    if 'segmentation' not in outputs:
        l = segmentation_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        segmentation_output.base_path = os.path.join(output_path, obj_fname, 'segmentation_output')
    
    if 'depth_01' not in outputs:
        l = depth_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        depth_output.base_path = os.path.join(output_path, obj_fname, 'depth_output')

    if 'normal' not in outputs:
        l = normal_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        normal_output.base_path = os.path.join(output_path, obj_fname, 'normal_output')

    if 'albedo' not in outputs:
        l = albedo_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        albedo_output.base_path = os.path.join(output_path, obj_fname, 'albedo_output')
    
    if 'depth_absolute' not in outputs:
        l = exr_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        exr_output.base_path = os.path.join(output_path, obj_fname, 'openEXR_output')

def set_output_paths_ShapeNet(tree, output_path, obj_synset, obj_fname, outputs):

    segmentation_output = tree.nodes['Segmentation']
    image_output = tree.nodes['Image_Output']
    depth_output = tree.nodes['Depth_Output']
    normal_output = tree.nodes['Normal_Output']
    albedo_output = tree.nodes['Albedo_Output']
    exr_output = tree.nodes['OpenEXR_Output']

    if 'image' not in outputs:
        l = image_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        image_output.base_path = os.path.join(output_path,
                obj_synset, obj_fname, 'image_output')

    if 'segmentation' not in outputs:
        l = segmentation_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        segmentation_output.base_path = os.path.join(output_path, 
                obj_synset,  obj_fname, 'segmentation_output')
    
    if 'depth_01' not in outputs:
        l = depth_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        depth_output.base_path = os.path.join(output_path, 
                obj_synset, obj_fname, 'depth_output')

    if 'normal' not in outputs:
        l = normal_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        normal_output.base_path = os.path.join(output_path, 
                obj_synset, obj_fname, 'normal_output')

    if 'albedo' not in outputs:
        l = albedo_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        albedo_output.base_path = os.path.join(output_path, 
                obj_synset, obj_fname, 'albedo_output')
    
    if 'depth_absolute' not in outputs:
        l = exr_output.inputs[0].links[0]
        tree.links.remove(l)
    else:
        exr_output.base_path = os.path.join(output_path, 
                obj_synset, obj_fname, 'openEXR_output')

def add_empty(coordinate, obj_name = 'constraint_obj'):
    x, y, z = coordinate
    
    obj = bpy.data.objects.new("Empty", None)
    obj.location = coordinate
    obj.name = obj_name

    return obj

def get_camera_location(r, azim, elev):
    
    azim = np.radians(-1*azim)
    elev = np.radians(elev)

    x = r*np.sin(np.pi/2-elev)*np.cos(azim)
    y = r*np.sin(np.pi/2-elev)*np.sin(azim)
    z = r*np.cos(np.pi/2-elev)

    return (x,y,z)

def add_lambertian_material(data):

    data.materials.new('lambertian')
    mat = bpy.data.materials['lambertian']
    mat.use_nodes = True

    return mat

def add_principled_material(data):

    data.materials.new('principled')
    mat = bpy.data.materials['principled']
    mat.use_nodes = True
   
    h,s,l = np.random.rand(), 0.5 + np.random.rand()/2.0, 0.4 + np.random.rand()/5.0
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]

    tree = mat.node_tree
    nodes = tree.nodes
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    
    nodes.remove(nodes['Diffuse BSDF'])
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (r, g, b, 1)

    BSDF.inputs['Specular'].default_value = np.random.uniform(0.6,0.9)
    BSDF.inputs['Roughness'].default_value = np.random.uniform(0.1,0.25)
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)

    return mat

def load_json(path):
    
    with open(path, 'r') as load_file:
        dct = json.load(load_file)        

    return dct

def load_materials(path):

    # load materials from blendfiel at <path> #
    with bpy.data.libraries.load(path) as (data_from, data_to):
            data_to.materials = [material for material in data_from.materials]

    # returna string with current material names #
    materials = [mat.name for mat in bpy.data.materials]
    return materials

def assign_material(obj, material_name):

    for slot in obj.material_slots:
            slot.material = bpy.data.materials[material_name].copy()

def adjust_links(tree):

    scene_image_output = tree.nodes['Render Layers'].outputs['Image']
    normal_output_input = tree.nodes['Normal_Output'].inputs['Image']

    output_nodes = [x for x in tree.nodes if 'Output' in x.name]
    for node in output_nodes:
        l = node.inputs[0].links[0]
        tree.links.remove(l)

    tree.links.new(scene_image_output, normal_output_input)

def remove_materials():
    # remove all materials in current project #
    materials = [mat.name for mat in bpy.data.materials]
    for material in materials:
        bpy.data.materials.remove(bpy.data.materials[material])
