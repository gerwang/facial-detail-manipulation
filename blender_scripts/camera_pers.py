## author: Sireer Wang (sireer.github.com)
## ${Blender_Path}/blender --background --factory-startup --python ./camera.py -- -m=./6_0_sRT.obj -s=./temp

import math
import os
import pickle

import bpy
import numpy as np
from scipy.spatial.transform import Rotation as R


class BlendshapeReader:
    def __init__(self, input_path):
        with open(input_path) as f:
            res = self.parse_file(f)
        self.bs = np.array(res)

    def parse_file(self, f):
        res = []
        for line in f.readlines():
            res.append(self.parse_line(line))
        return res

    def clip(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.bs)
        self.bs = self.bs[start:end]

    def parse_line(self, line):
        line = line.split()
        index = 0
        while line[index] != 'C':
            index += 1
        index += 1
        num = int(line[index])
        index += 1
        res = list(map(float, line[index:index + num]))
        return res


class Camera(object):
    def __init__(self, fx, fy, cx, cy, height, width):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.height = height
        self.width = width
        self.blender_camera = BlenderCamera(self)

    def project(self, v):
        return np.asarray([self.fx * v[0] / v[2] + self.cx, self.fy * v[1] / v[2] + self.cy])

    def blender_project(self, v):
        return np.asarray(
            [self.blender_camera.scale_width, self.blender_camera.scale_height]) * self.blender_camera.project(v)


class BlenderCamera(object):
    """Blender Camera
    """

    def __init__(self, camera: Camera):
        self.height = camera.height * camera.fx / camera.fy
        self.width = camera.width
        self.size = max(self.height, self.width)

        self.scale_height = camera.fy / camera.fx
        self.scale_width = 1.0

        self.focal_length = 50.0
        self.F = camera.fx
        self.f = self.F / self.size
        self.sensor_size = self.focal_length * self.size / self.F
        self.sensor_width = self.sensor_size * self.width / self.size
        self.sensor_height = self.sensor_size * self.height / self.size

        self.shift_x = - (camera.cx / camera.fx - self.width / self.F * 0.5) * self.f
        self.shift_y = (camera.cy / camera.fy - self.height / self.F * 0.5) * self.f
        # self.shift_x = (camera.cx - self.width * 0.5) / self.size
        # self.shift_y = (camera.cy - self.height * 0.5) / self.size

    def project(self, v):
        return self.size * (self.f * np.asarray([v[0] / v[2], v[1] / v[2]]) - np.asarray(
            [self.shift_x, self.shift_y])) + np.asarray([self.width * 0.5, self.height * 0.5])


def enable_cuda():
    ## this function if borrowed from https://github.com/nytimes/rd-blender-docker/issues/3#issuecomment-618459326
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    # Calling this purges the device list so we need it
    cprefs.refresh_devices()
    cuda_devices, opencl_devices = cprefs.devices[:2]
    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL'):
        try:
            cprefs.compute_device_type = compute_device_type
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        device.use = True


def load_parser():
    import sys
    import argparse  # to parse options for us and print a nice help message
    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments

    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # When --help or no args are given, print this help
    usage_text = (
            "Run blender in background mode with this script:"
            "  blender --background --python " + __file__ + " -- [options]"
    )
    parser = argparse.ArgumentParser(description=usage_text)

    # Example utility, add some text and renders or saves it (with options)
    # Possible types are: string, int, long, choice, float and complex.
    parser.add_argument(
        "-m", "--mesh", dest="mesh", type=str, required=True,
        help="This mesh will be used to render an image",
    )
    parser.add_argument(
        "-d", "--disp",
        help="This displacement map will be used to render an image",
    )
    parser.add_argument(
        '-d2', '--disp_2',
        help='Target displacement to interpolate'
    )
    parser.add_argument(
        "-s", "--save", dest="save_path", metavar='FILE',
        help="Save the generated file to the specified path",
    )
    parser.add_argument(
        "-r", "--render", dest="render_path", metavar='FILE',
        help="Render an image to the specified path",
    )
    parser.add_argument(
        '-p', '--param', metavar='FILE', help='File to read pose parameters'
    )
    parser.add_argument(
        '-b', '--background', metavar='FILE', help='background image path'
    )
    parser.add_argument('-m2', '--mesh_2', help='The destination mesh for morphing animation')
    parser.add_argument('-f', '--frame', type=int, default=30, help='Number of frames for linear interpolation')
    parser.add_argument('--is_bs', action='store_true', help='True if loading a directory of blendshape meshes')
    parser.add_argument('-nb', '--no_back', action='store_true',
                        help='Set this if use background image to determine render size, but do not show background image')
    parser.add_argument('-sm', '--smooth', action='store_true', help='Apply smooth before rendering')
    parser.add_argument('-dd', '--dynamic_d', action='store_true', help='Use dynamic displacement map for animation')
    parser.add_argument('-sf', '--skip_frame', type=int, default=0,
                        help='Skip several frames of dynamic displacement map')
    parser.add_argument('-bs', '--blendshape', help='blendshape coeff path')
    parser.add_argument('-ds', '--dpmap_scale', type=float, default=4.0)
    parser.add_argument('-t', '--texture')
    parser.add_argument('--rot', type=float, default=0.0)
    parser.add_argument('-e', '--engine', choices=['e', 'c'], default='e')
    parser.add_argument('-em', '--env', default='HDR_112_River_Road_2_Env.hdr')
    parser.add_argument('--em_rot', type=float, default=90, help='Environment map rotation angle')
    parser.add_argument('--samples', type=int, default=64, help='Cycles render sample number')
    parser.add_argument('-ls', '--light_strength', type=float, default=1.0)
    parser.add_argument('--no_denoise', action='store_true')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--img_type', choices=['jpg', 'png'], default='jpg')
    parser.add_argument('-wp', '--weak-perspective', action='store_true')
    parser.add_argument('--bs_clip_start', type=int, default=None)
    parser.add_argument('--bs_clip_end', type=int, default=None)

    args = parser.parse_args(argv)  # In this example we won't use the args

    if not argv:
        parser.print_help()
        return

    if not args.mesh:
        print("Error: --mesh=\"some string\" argument not given, aborting.")
        parser.print_help()
        return
    return args


def setup_camera():
    bpy.context.scene.use_nodes = True
    composite_input = bpy.context.scene.node_tree.nodes['Composite'].inputs[0]

    if args.img_type == 'jpg':
        node_render = bpy.context.scene.node_tree.nodes['Render Layers']
        node_alpha = bpy.context.scene.node_tree.nodes.new('CompositorNodeAlphaOver')
        compsiting_link = bpy.context.scene.node_tree.links
        compsiting_link.new(node_render.outputs[0], node_alpha.inputs[2])
        compsiting_link.new(node_alpha.outputs[0], composite_input)
        composite_input = node_alpha.inputs[2]

    if args.background is not None:
        filepath = args.background

        img = bpy.data.images.load(filepath)
        image_width, image_height = img.size

        if not args.no_back:
            node_render = bpy.context.scene.node_tree.nodes['Render Layers']
            node_alpha = bpy.context.scene.node_tree.nodes.new('CompositorNodeAlphaOver')
            node_image = bpy.context.scene.node_tree.nodes.new('CompositorNodeImage')
            node_image.image = img

            compositing_link = bpy.context.scene.node_tree.links
            compositing_link.new(node_image.outputs[0], node_alpha.inputs[1])
            compositing_link.new(node_render.outputs[0], node_alpha.inputs[2])
            compositing_link.new(node_render.outputs[1], node_alpha.inputs[0])
            compositing_link.new(node_alpha.outputs[0], composite_input)
            composite_input = node_alpha.inputs[2]
    else:
        image_width = args.size
        image_height = args.size

    cam_data = bpy.data.cameras.new("MyCam")
    cam_ob = bpy.data.objects.new(name="MyCam", object_data=cam_data)
    cam_ob.parent = parent

    # assume image is not scaled
    assert bpy.context.scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view

    # instance the camera object in the scene
    bpy.context.scene.collection.objects.link(cam_ob)
    bpy.context.scene.camera = cam_ob

    if args.weak_perspective:
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.type = 'ORTHO'
        cam_data.shift_x = 0
        cam_data.shift_y = 0
        cam_data.clip_start = 1
        cam_data.clip_end = 10000
        cam_ob.location = image_width / 2, image_height / 2, 1000
        cam_ob.rotation_euler = 0, 0, 0
        cam_data.ortho_scale = image_width
        bpy.context.scene.render.resolution_x = image_width
        bpy.context.scene.render.resolution_y = image_height
    else:
        cam_data.sensor_fit = 'VERTICAL'
        camera = Camera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=image_width, height=image_height)
        cam_data.shift_x = camera.blender_camera.shift_x
        cam_data.shift_y = camera.blender_camera.shift_y
        cam_data.sensor_width = camera.blender_camera.sensor_width
        cam_data.sensor_height = camera.blender_camera.sensor_height
        cam_data.lens = camera.blender_camera.focal_length
        cam_ob.location = 0.0, 0.0, 0.0
        cam_ob.rotation_euler[0] = math.pi
        cam_ob.rotation_euler[1] = 0
        cam_ob.rotation_euler[2] = 0
        bpy.context.scene.render.resolution_x = int(camera.blender_camera.width + 0.5)
        bpy.context.scene.render.resolution_y = int(camera.blender_camera.height + 0.5)

    bpy.context.scene.render.film_transparent = True  # set background to transparent
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '16'

    if args.engine == 'e':
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    else:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = args.samples
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'

        # enable denoise
        if not args.no_denoise:
            bpy.context.scene.use_nodes = True
            node_render = bpy.context.scene.node_tree.nodes['Render Layers']
            node_denoise = bpy.context.scene.node_tree.nodes.new(
                "CompositorNodeDenoise")
            compsiting_link = bpy.context.scene.node_tree.links
            compsiting_link.new(node_render.outputs[0], node_denoise.inputs[0])
            compsiting_link.new(node_denoise.outputs[0], composite_input)
            composite_input = node_denoise.inputs[0]

    bpy.context.scene.display_settings.display_device = 'None'

    if args.env is not None and args.engine == 'c':
        # add env map & rotate it
        if bpy.context.scene.world is None:
            # create a new world
            new_world = bpy.data.worlds.new("New World")
            bpy.context.scene.world = new_world
        world = bpy.context.scene.world
        world.use_nodes = True
        world_links = world.node_tree.links
        node_tex_coord = world.node_tree.nodes.new("ShaderNodeTexCoord")
        node_mapping = world.node_tree.nodes.new("ShaderNodeMapping")
        node_mapping.inputs[2].default_value = 0, 0, args.em_rot / 180 * math.pi
        node_env_texture = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        node_env_texture.image = bpy.data.images.load(filepath=args.env)
        node_env_texture.image.colorspace_settings.name = 'Raw'
        node_color2bw = world.node_tree.nodes.new('ShaderNodeRGBToBW')
        world_links.new(node_tex_coord.outputs[0], node_mapping.inputs[0])
        world_links.new(node_mapping.outputs[0], node_env_texture.inputs[0])
        world_links.new(node_env_texture.outputs[0], node_color2bw.inputs[0])
        world_links.new(node_color2bw.outputs[0], world.node_tree.nodes['Background'].inputs[0])
        world.node_tree.nodes['Background'].inputs[1].default_value = args.light_strength

    if args.background is not None:
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.image = img


def setup_weak_perspective_camera():
    if args.background is not None:
        filepath = args.background

        img = bpy.data.images.load(filepath)
        image_width, image_height = img.size

        bpy.context.scene.use_nodes = True

        for n in bpy.context.scene.node_tree.links:
            bpy.context.scene.node_tree.links.remove(n)

        node_Composite = bpy.context.scene.node_tree.nodes['Composite']
        node_render = bpy.context.scene.node_tree.nodes['Render Layers']
        node_alpha = bpy.context.scene.node_tree.nodes.new('CompositorNodeAlphaOver')
        node_image = bpy.context.scene.node_tree.nodes.new('CompositorNodeImage')
        node_image.image = img

        compositing_link = bpy.context.scene.node_tree.links
        compositing_link.new(node_image.outputs[0], node_alpha.inputs[1])
        compositing_link.new(node_render.outputs[0], node_alpha.inputs[2])
        compositing_link.new(node_render.outputs[1], node_alpha.inputs[0])
        compositing_link.new(node_alpha.outputs[0], node_Composite.inputs[0])
    else:
        image_width = args.size
        image_height = args.size

    cam_data = bpy.data.cameras.new("MyCam")
    cam_ob = bpy.data.objects.new(name="MyCam", object_data=cam_data)

    cam_data.sensor_fit = 'HORIZONTAL'
    # instance the camera object in the scene
    bpy.context.scene.collection.objects.link(cam_ob)
    bpy.context.scene.camera = cam_ob

    cam_data.type = 'ORTHO'
    cam_data.shift_x = 0
    cam_data.shift_y = 0
    cam_data.clip_start = 1
    cam_data.clip_end = 10000
    cam_ob.location = image_width / 2, image_height / 2, 1000
    cam_ob.rotation_euler = 0, 0, 0
    cam_data.ortho_scale = image_width

    bpy.context.scene.render.resolution_x = image_width
    bpy.context.scene.render.resolution_y = image_height
    bpy.context.scene.render.film_transparent = True  # set background to transparent
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '16'

    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.display_settings.display_device = 'None'

    if args.background is not None:
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.image = img


def setup_lights_down():
    area_light_data = bpy.data.lights.new('Area_Light', type='AREA')
    area_light_ob = bpy.data.objects.new(name='Area_Light', object_data=area_light_data)
    bpy.context.scene.collection.objects.link(area_light_ob)
    area_light_ob.location = 0.799123, 1.121831, 1.96994
    area_light_ob.rotation_euler = 246.365 / 180 * math.pi, -12.4575 / 180 * math.pi, -28.3528 / 180 * math.pi
    area_light_ob.parent = parent
    area_light_data.energy = 8
    area_light_data.size = 1

    area_light_data = bpy.data.lights.new('Area_Light_2', type='AREA')
    area_light_ob = bpy.data.objects.new(name='Area_Light_2', object_data=area_light_data)
    bpy.context.scene.collection.objects.link(area_light_ob)
    area_light_ob.location = -0.75937, 1.11015, 1.98205
    area_light_ob.rotation_euler = 247.625 / 180 * math.pi, 12.2662 / 180 * math.pi, 28.7615 / 180 * math.pi
    area_light_ob.parent = parent
    area_light_data.energy = 8
    area_light_data.size = 1


def setup_lights():
    area_light_data = bpy.data.lights.new("Area_Light", type='AREA')
    area_light_2_data = bpy.data.lights.new("Area_Light", type='POINT')
    area_light_ob = bpy.data.objects.new(name="Area_Light", object_data=area_light_data)
    area_light_2_ob = bpy.data.objects.new(name="Area_Light", object_data=area_light_2_data)
    bpy.context.scene.collection.objects.link(area_light_ob)
    bpy.context.scene.collection.objects.link(area_light_2_ob)

    area_light_data.type = 'AREA'
    area_light_data.energy = 80
    area_light_ob.location = 2.70953, 0, -0.194689
    area_light_ob.rotation_euler[0] = 0
    area_light_ob.rotation_euler[1] = 98.2077 / 180 * math.pi
    area_light_ob.rotation_euler[2] = 0
    area_light_ob.scale[0] = 5
    area_light_ob.scale[1] = 5
    area_light_ob.scale[2] = 5

    area_light_2_data.type = 'AREA'
    area_light_2_data.energy = 12
    area_light_2_ob.location = -0.721643, -0.348582, -0.258723
    area_light_2_ob.rotation_euler[0] = 17.6306 / 180 * math.pi
    area_light_2_ob.rotation_euler[1] = 228.631 / 180 * math.pi
    area_light_2_ob.rotation_euler[2] = 0.03223 / 180 * math.pi
    area_light_2_ob.scale[0] = 2.15177
    area_light_2_ob.scale[1] = 2.15177
    area_light_2_ob.scale[2] = 2.15177


def load_mesh(obj_path):
    if obj_path.lower().endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=obj_path, split_mode='OFF')
    else:
        raise ValueError(f'{os.path.splitext(obj_path)[1]} not supported')
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    # bpy.data.objects[obj_name].select_set(True)
    obj = bpy.data.objects[obj_name]
    bpy.context.view_layer.objects.active = obj
    print(bpy.ops.object.shade_smooth())
    obj.rotation_euler[0] = 0
    obj.rotation_euler[1] = 0
    obj.rotation_euler[2] = 0
    # print(bpy.data.meshes[obj_name].materials[0].node_tree.nodes[0].input.keys())
    bpy.data.meshes[obj_name].materials[0].node_tree.nodes[0].inputs[5].default_value = 0.2  # specular
    bpy.data.meshes[obj_name].materials[0].node_tree.nodes[0].inputs[7].default_value = 0.5  # roughness
    bpy.data.meshes[obj_name].materials[0].node_tree.nodes[0].inputs[0].default_value = 0.5, 0.5, 0.8, 1  # roughness

    if args.smooth:
        mod = obj.modifiers.new('LaplacianSmooth', 'LAPLACIANSMOOTH')
        mod.iterations = 3
        mod.lambda_factor = 0.5
        mod.lambda_border = 0.3
        print(bpy.ops.object.modifier_apply(modifier=mod.name))

    obj.parent = parent
    return obj_name


def setup_texture():
    global node_mix, node_dis_tex, node_dis_tex_2
    node_tree = bpy.data.meshes[obj_name].materials[0].node_tree

    if args.disp is not None:
        node_dis_tex = node_tree.nodes.new(type="ShaderNodeTexImage")
        if args.dynamic_d:
            node_dis_tex.image = bpy.data.images.load(filepath=args.disp.format(0))
        else:
            node_dis_tex.image = bpy.data.images.load(filepath=args.disp)
        node_dis_tex.image.colorspace_settings.name = 'Non-Color'
        node_dis_tex.interpolation = 'Cubic'
        node_dis_tex.extension = 'EXTEND'

        if args.disp_2 is not None or args.skip_frame > 0:
            node_dis_tex_2 = node_tree.nodes.new(type="ShaderNodeTexImage")
            if args.disp_2 is not None:
                node_dis_tex_2.image = bpy.data.images.load(filepath=args.disp_2)
            else:
                node_dis_tex_2.image = bpy.data.images.load(filepath=args.disp.format(args.skip_frame + 1))
            node_dis_tex_2.image.colorspace_settings.name = 'Non-Color'
            node_dis_tex_2.interpolation = 'Cubic'
            node_dis_tex_2.extension = 'EXTEND'

        node_vector_dis = node_tree.nodes.new(type="ShaderNodeDisplacement")
        node_vector_dis.inputs[2].default_value = args.dpmap_scale  # scale
        node_material_output = node_tree.nodes['Material Output']

        objects_link = node_tree.links
        if args.disp_2 is not None or args.skip_frame > 0:
            node_mix = node_tree.nodes.new(type='ShaderNodeMixRGB')
            node_mix.inputs[0].default_value = 0.0
            objects_link.new(node_dis_tex.outputs[0], node_mix.inputs[1])
            objects_link.new(node_dis_tex_2.outputs[0], node_mix.inputs[2])
            objects_link.new(node_mix.outputs[0], node_vector_dis.inputs[0])
            objects_link.new(node_vector_dis.outputs[0], node_material_output.inputs[2])
        else:
            objects_link.new(node_dis_tex.outputs[0], node_vector_dis.inputs[0])
            objects_link.new(node_vector_dis.outputs[0], node_material_output.inputs[2])


def setup_normal_texture():
    bpy.context.scene.use_nodes = True
    node_normal = bpy.data.meshes[obj_name].materials[0].node_tree.nodes.new("ShaderNodeTexCoord")
    node_normal_multiply = bpy.data.meshes[obj_name].materials[0].node_tree.nodes.new("ShaderNodeVectorMath")
    node_normal_add = bpy.data.meshes[obj_name].materials[0].node_tree.nodes.new("ShaderNodeVectorMath")

    node_normal_multiply.operation = 'MULTIPLY'
    if args.disp is None:
        node_normal_multiply.inputs[1].default_value = 0.5, 0.5, 0.5
    else:
        node_normal_multiply.inputs[1].default_value = 0.5, -0.5, -0.5
    node_normal_add.operation = 'ADD'
    node_normal_add.inputs[1].default_value[0] = 0.5
    node_normal_add.inputs[1].default_value[1] = 0.5
    node_normal_add.inputs[1].default_value[2] = 0.5
    objects_link = bpy.data.meshes[obj_name].materials[0].node_tree.links
    objects_link.new(node_normal.outputs[1], node_normal_multiply.inputs[0])
    objects_link.new(node_normal_multiply.outputs[0], node_normal_add.inputs[0])
    node_material_output = bpy.data.meshes[obj_name].materials[0].node_tree.nodes['Material Output']
    if args.texture is None:
        objects_link.new(node_normal_add.outputs[0], node_material_output.inputs[0])
    else:
        node_img_tex = bpy.data.meshes[obj_name].materials[0].node_tree.nodes.new('ShaderNodeTexImage')
        node_img_tex.image = bpy.data.images.load(filepath=args.texture)
        node_mix = bpy.data.meshes[obj_name].materials[0].node_tree.nodes.new('ShaderNodeMixRGB')
        objects_link.new(node_img_tex.outputs[1], node_mix.inputs[0])
        objects_link.new(node_normal_add.outputs[0], node_mix.inputs[1])
        objects_link.new(node_img_tex.outputs[0], node_mix.inputs[2])
        objects_link.new(node_mix.outputs[0], node_material_output.inputs[0])


def render(suffix=''):
    if args.engine == 'c':
        if args.weak_perspective:
            parent.rotation_euler[0] = math.pi / 2
        else:
            parent.rotation_euler[0] = -math.pi / 2
    if args.img_type == 'jpg':
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.context.scene.render.image_settings.quality = 90
    bpy.context.scene.render.filepath = args.save_path + suffix
    bpy.ops.render.render(write_still=True)
    # obj.rotation_euler[1] = np.pi / 4
    # bpy.context.scene.render.filepath = args.save_path + '_left'
    # bpy.ops.render.render(write_still=True)
    # obj.rotation_euler[1] = 0
    # obj.rotation_euler = np.pi, 0, 0
    # obj.location[0] = 0
    # obj.rotation_euler = 166 / 180 * np.pi, 0, 0
    # obj.location = 0.066601, 0.153837, 1.30025
    # bpy.context.scene.render.filepath = args.save_path + '_middle'
    # bpy.ops.render.render(write_still=True)
    # obj.rotation_euler[1] = -np.pi / 4
    # bpy.context.scene.render.filepath = args.save_path + '_right'
    # bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    args = load_parser()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    parent = bpy.data.objects.new(name="Parent", object_data=None)
    bpy.context.scene.collection.objects.link(parent)

    # setup_lights()
    setup_lights_down()
    obj_name = load_mesh(args.mesh)
    obj = bpy.data.objects[obj_name]
    if args.param is not None:
        if args.param.endswith('.pkl'):
            params = pickle.load(open(args.param, 'rb'))
        elif args.param.endswith('.npz'):
            params = np.load(args.param)
        else:
            raise ValueError(f'unknown {args.param}')
        obj.scale = [params['scale'] for _ in range(3)]
        obj.rotation_euler = R.from_rotvec(params['rot_vector']).as_euler('xyz').tolist()
        if args.weak_perspective:
            obj.location = params['trans'].tolist() + [0]
            K = None
        else:
            obj.location = params['trans'].tolist()
            K = params['K']
    else:
        obj.scale = [0.003 for _ in range(3)]
        obj.rotation_euler = math.pi, args.rot / 180 * math.pi, 0
        obj.location = 0, 0, 2.5
        K = np.array([
            [3650 / 1024 * args.size, 0, 512 / 1024 * args.size],
            [0, 3650 / 1024 * args.size, 512 / 1024 * args.size],
            [0, 0, 1]
        ])

    setup_camera()
    node_mix = None
    node_dis_tex = None
    node_dis_tex_2 = None
    setup_texture()
    if args.engine == 'e':
        setup_normal_texture()
    enable_cuda()
    if args.is_bs:
        n_bs = 52
        shapes = [obj]
        for i in range(1, n_bs):
            mesh_name = load_mesh(args.mesh.replace('0.obj', f'{i}.obj'))
            shapes.append(bpy.data.objects[mesh_name])
        bpy.context.view_layer.objects.active = obj
        for x in shapes:
            x.select_set(True)
        print(bpy.ops.object.join_shapes())
        for x in shapes[1:]:
            x.hide_viewport = True
            x.hide_render = True
        if args.blendshape is not None:  # render blendshape anime
            bs_reader = BlendshapeReader(args.blendshape)
            bs_reader.clip(args.bs_clip_start, args.bs_clip_end)
            for j in range(len(bs_reader.bs)):
                for i in range(1, len(obj.data.shape_keys.key_blocks)):
                    obj.data.shape_keys.key_blocks[i].value = bs_reader.bs[j, i - 1]
                if args.dynamic_d:
                    node_dis_tex.image = bpy.data.images.load(filepath=args.disp.format(j))
                    node_dis_tex.image.colorspace_settings.name = 'Non-Color'
                render(f'_{j}')
        else:
            for i in range(len(obj.data.shape_keys.key_blocks)):
                for j in range(len(obj.data.shape_keys.key_blocks)):
                    obj.data.shape_keys.key_blocks[j].value = 0
                obj.data.shape_keys.key_blocks[i].value = 1
                render(f'_{i}')
    elif args.mesh_2 is None and node_mix is None and not args.dynamic_d:
        render()
    else:
        if args.mesh_2 is not None:
            target_obj_name = load_mesh(args.mesh_2)
            target_obj = bpy.data.objects[target_obj_name]
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            target_obj.select_set(True)
            print(bpy.ops.object.join_shapes())
            target_obj.hide_viewport = True
            target_obj.hide_render = True
        cur_start = -1
        cur_end = -1
        for i in range(args.frame):
            if args.mesh_2 is not None:
                obj.data.shape_keys.key_blocks[1].value = i / (args.frame - 1)
            if args.skip_frame > 0:
                local_progress = i % (args.skip_frame + 1)
                if local_progress == 0:
                    cur_start = i
                    cur_end = min(i + args.skip_frame + 1, args.frame - 1)
                    node_dis_tex.image = bpy.data.images.load(filepath=args.disp.format(cur_start))
                    node_dis_tex.image.colorspace_settings.name = 'Non-Color'
                    node_dis_tex_2.image = bpy.data.images.load(filepath=args.disp.format(cur_end))
                    node_dis_tex_2.image.colorspace_settings.name = 'Non-Color'
                node_mix.inputs[0].default_value = 0 if cur_start == cur_end else (i - cur_start) / (
                        cur_end - cur_start)
            else:
                if node_mix is not None:
                    node_mix.inputs[0].default_value = i / (args.frame - 1)
                if args.dynamic_d:
                    node_dis_tex.image = bpy.data.images.load(filepath=args.disp.format(i))
                    node_dis_tex.image.colorspace_settings.name = 'Non-Color'
            render(suffix=f'_{i}')
    '''
    bpy.ops.mesh.uv_texture_add()
    bpy.context.object.data.uv_layers["UVMap.001"].active_render = True
    bpy.ops.uv.project_from_view(camera_bounds=True, correct_aspect=False, scale_to_bounds=False)
    bpy.context.scene.tool_settings.image_paint.mode = 'IMAGE'
    bpy.ops.image.new(name="Untitled")
    bpy.ops.wm.context_set_int(data_path="active_object.data.uv_layers.active_index", value=0)
    '''
