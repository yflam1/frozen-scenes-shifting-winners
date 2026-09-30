import math
import os
import sys
import warnings
from contextlib import contextmanager
from copy import deepcopy
from math import atan, radians, sin, tan
from os import makedirs
from os.path import dirname, exists
from typing import (
    Iterator,
    Literal,
    Optional,
    TextIO,
    TypeVar,
    Union,
    overload,
)

# isort: off

import bpy  # type: ignore
import bmesh  # type: ignore
from bpy.types import Image, Object  # type: ignore
from mathutils import Euler, Matrix, Vector  # type: ignore

# isort: on

DEFAULT_ENVIRONMENT_STRENGTH = 1.0
DEFAULT_HEIGHT_MIDLEVEL = 0.5
DEFAULT_HEIGHT_SCALE = 0.01
DEFAULT_EMISSION_STRENGTH = 1.0

T = TypeVar("T")
PyVector3D = tuple[float, float, float]
ColorRGB = tuple[int, int, int]
Strength = float
Midlevel = float
Scale = float
DegreeCCW = float


@contextmanager
def redirect_stdout(to: str = os.devnull) -> Iterator[None]:
    """
    Reference: https://blender.stackexchange.com/a/270199
    """

    fd = sys.stdout.fileno()

    def _redirect_stdout(to: TextIO):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to `to` file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(file)
        try:
            yield
        finally:
            _redirect_stdout(old_stdout)


def clear() -> None:
    # Delete scene objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove datablocks that otherwise become orphans
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
        bpy.data.collections,
    ):
        for block in list(coll):
            coll.remove(block, do_unlink=True)

    # Purge anything still orphaned
    with redirect_stdout():
        bpy.ops.outliner.orphans_purge(do_recursive=True)


def duplicate(obj: Object, name: Optional[str] = None) -> Object:
    if name is None:
        name = f"{obj.name}_copy"
    focus(obj)
    bpy.ops.object.duplicate(linked=False)
    obj_copy = bpy.context.active_object
    obj_copy.name = name
    focus(obj_copy)
    return obj_copy


def export_obj(
    path: str,
    obj: Optional[Object] = None,
    export_vertex_colors: bool = False,
    export_materials: bool = True,
) -> bool:
    export_dir = dirname(path)
    if export_dir != "" and not exists(export_dir):
        makedirs(export_dir)
    if obj is not None:
        # Store original visibility
        visibility = {o: o.hide_get() for o in bpy.context.scene.objects}
        # Make only the target object visible
        for o in bpy.context.scene.objects:
            o.hide_set(True if o != obj else False)
    with redirect_stdout():
        bpy.ops.wm.obj_export(
            filepath=path,
            export_colors=export_vertex_colors,
            export_materials=export_materials,
            export_pbr_extensions=True,
            path_mode="COPY",
        )
    if obj is not None:
        # Restore original visibility
        for o, was_hidden in visibility.items():
            o.hide_set(was_hidden)
    return exists(path)


def focus(obj: Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def initialize(
    *,
    transparent: Optional[bool] = False,
    environment_map: Optional[tuple[str, Strength]] = None,
    color_depth: Literal["8", "16"] = "8",
    compression: int = 15,
    samples: Optional[int] = None,
    use_adaptive_sampling: bool = True,
    adaptive_threshold: float = 0.02,
    use_denoising: bool = True,
    use_light_tree: bool = True,
    pixel_filter_type: Literal[
        "BOX", "GAUSSIAN", "BLACKMAN_HARRIS"
    ] = "BLACKMAN_HARRIS",
    filter_width: float = 0.1,
    exposure: Optional[float] = None,
) -> None:
    """
    Args:
        compression (int, optional): Lower values output faster with less
            lossless compression. Defaults to 15.
        samples (Optional[int], optional): Number of samples to render for
            each pixel. Defaults to 64.
        use_adaptive_sampling (bool, optional): Automatically reduce
            the number of samples per pixel based on estimated noise level.
            Defaults to True.
        adaptive_threshold (float, optional): Lower values reduce
            noise at the cost of render time. Defaults to 0.02.
        filter_width (float, optional): Lower values give more
            aliased but less blurry edges. Defaults to 0.1.
        use_light_tree: Use light tree for faster light sampling.
            Defaults to False.
    """

    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences

    # Reference: https://blender.stackexchange.com/a/196702
    # Priority order: OptiX (RTX) > CUDA (NVIDIA) > HIP (AMD) > METAL (Apple) > CPU
    # OptiX leverages RT cores on RTX cards for hardware-accelerated ray tracing
    selected_device_type = "NONE"
    for device_type in ["OPTIX", "CUDA", "HIP", "METAL", "NONE"]:
        try:
            cycles_prefs.compute_device_type = device_type
            selected_device_type = device_type
            break
        except TypeError:
            continue

    # Refresh device list after setting compute device type
    cycles_prefs.get_devices()

    # Enable GPU devices and DISABLE CPU for pure GPU rendering
    # Hybrid CPU+GPU rendering often causes performance degradation
    gpu_available = False
    for device in cycles_prefs.devices:
        if device.type == "CPU":
            device.use = False  # Explicitly disable CPU for GPU rendering
        else:
            device.use = True
            gpu_available = True

    # Fallback to CPU if no GPU is available
    if not gpu_available:
        for device in cycles_prefs.devices:
            if device.type == "CPU":
                device.use = True
                break
        else:
            raise RuntimeError("No GPU found, and CPU is not available")

    # Set background color (or environment map)
    bpy.ops.world.new()
    world = bpy.context.scene.world = bpy.data.worlds[-1]
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    if environment_map is not None:
        if not isinstance(environment_map, tuple) or len(environment_map) != 2:
            raise ValueError("`environment_map` must be (path, strength)")
        e1, e2 = environment_map
        if not exists(e1):
            raise FileNotFoundError(f"Environment map not found: {e1}")
        e2 = max(e2, 0.0)
        world.node_tree.nodes["Background"].inputs[1].default_value = e2
        env_map_node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env_map_node.image = bpy.data.images.load(e1)
        world.node_tree.links.new(
            env_map_node.outputs[0],
            world.node_tree.nodes["Background"].inputs[0],
        )

    # ==========================================================================
    # CYCLES RENDER ENGINE CONFIGURATION
    # ==========================================================================

    context = bpy.context
    scene = context.scene
    scene.view_settings.view_transform = "AgX"

    cycles = scene.cycles
    cycles.device = "GPU" if gpu_available else "CPU"

    # --- Sampling Settings ---
    cycles.samples = 64 if samples is None else max(1, samples)
    cycles.use_adaptive_sampling = use_adaptive_sampling
    cycles.adaptive_threshold = adaptive_threshold
    # Minimum samples before adaptive sampling kicks in (prevents early termination)
    cycles.adaptive_min_samples = 0  # 0 = auto (usually samples/4)

    # --- Denoising Settings (RTX Optimization) ---
    cycles.use_denoising = use_denoising
    if use_denoising:
        # OptiX denoiser uses Tensor cores on RTX GPUs - MUCH faster than OpenImageDenoise
        # OpenImageDenoise is better quality but runs on CPU
        if gpu_available and selected_device_type == "OPTIX":
            cycles.denoiser = "OPTIX"
        else:
            cycles.denoiser = "OPENIMAGEDENOISE"

        # More input passes = better denoising quality but more VRAM usage
        # RGB_ALBEDO_NORMAL is best quality, RGB is fastest
        cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

        # Denoise on GPU if possible (for OIDN with GPU support in Blender 4.0+)
        if hasattr(cycles, "denoising_use_gpu"):
            cycles.denoising_use_gpu = True

    # --- Light Sampling ---
    # Light tree (Blender 3.5+) dramatically improves performance with many lights
    if hasattr(cycles, "use_light_tree"):
        cycles.use_light_tree = use_light_tree

    # --- Pixel Filter ---
    cycles.pixel_filter_type = pixel_filter_type
    cycles.filter_width = filter_width

    # --- Exposure ---
    cycles.film_exposure = 1.0 if exposure is None else max(0.01, exposure)

    # Use spatial splits for better BVH (slower build, faster render)
    if hasattr(cycles, "debug_use_spatial_splits"):
        cycles.debug_use_spatial_splits = True

    # ==========================================================================
    # RENDER OUTPUT SETTINGS
    # ==========================================================================

    render = scene.render
    render.engine = "CYCLES"
    render.film_transparent = True if transparent is None else transparent
    render.resolution_percentage = 100
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0
    render.use_file_extension = True

    # Image format settings
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = color_depth
    render.image_settings.compression = compression

    # Change color space from Linear Rec.709
    bpy.ops.wm.set_working_color_space(working_space="Linear Rec.2020")


def is_mesh(obj: Object, raise_err: bool = False) -> bool:
    if obj.type == "MESH":
        return True
    if raise_err:
        raise TypeError(f"Object {obj.name} is not a mesh")
    return False


def import_obj(
    obj_path: str, load_vertex_colors: bool = False, use_shadow: bool = True
) -> Object:
    with redirect_stdout():
        bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]

    if load_vertex_colors and is_mesh(obj):
        colors = []
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    if len(parts) >= 7:
                        colors.append(tuple(map(float, parts[4:7])))
                    else:
                        colors.append((0.0, 0.0, 0.0))

        mesh = obj.data
        color_layer = mesh.vertex_colors.new(name="ImportedVertexColor")
        for face in mesh.polygons:
            for loop_idx in face.loop_indices:
                vert_idx = mesh.loops[loop_idx].vertex_index
                color_layer.data[loop_idx].color = colors[vert_idx] + (1.0,)

        mat = bpy.data.materials.new(name="BakedMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        links.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        vcol = nodes.new(type="ShaderNodeVertexColor")
        links.new(vcol.outputs["Color"], diffuse.inputs["Color"])
        links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    obj.visible_shadow = use_shadow
    return obj


def import_ply(
    ply_path: str, point_radius: float = 0.05, *, max_z: Optional[float] = None
) -> Object:
    with redirect_stdout():
        bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]

    # Shading
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    att = nodes.new("ShaderNodeAttribute")
    att.attribute_name = "Col"
    links.new(att.outputs["Color"], bsdf.inputs["Base Color"])

    # Geometry
    ng = bpy.data.node_groups.new(
        name="PointCloudGeometryNodeTree", type="GeometryNodeTree"
    )
    ng.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    ng.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    mod = obj.modifiers.new("PointCloudGeometryNodes", type="NODES")
    mod.node_group = ng
    ng_nodes = ng.nodes
    ng_links = ng.links
    group_in = ng_nodes.new("NodeGroupInput")
    group_out = ng_nodes.new("NodeGroupOutput")
    m2p = ng_nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = point_radius
    set_mat = ng_nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    ng_links.new(group_in.outputs["Geometry"], m2p.inputs["Mesh"])
    ng_links.new(m2p.outputs["Points"], set_mat.inputs["Geometry"])
    ng_links.new(set_mat.outputs["Geometry"], group_out.inputs["Geometry"])

    # Disable points above the desired height
    if max_z is not None:
        pos = ng_nodes.new("GeometryNodeInputPosition")
        sep = ng_nodes.new("ShaderNodeSeparateXYZ")
        com = ng_nodes.new("FunctionNodeCompare")
        com.operation = "LESS_THAN"
        com.inputs["B"].default_value = max_z
        ng_links.new(pos.outputs["Position"], sep.inputs["Vector"])
        ng_links.new(sep.outputs["Z"], com.inputs["A"])
        ng_links.new(com.outputs["Result"], m2p.inputs["Selection"])

    return obj


def load_blender_image(path: str) -> Image:
    return bpy.data.images.load(path, check_existing=True)


def transform(
    obj: Object,
    position: Optional[PyVector3D] = None,
    rotation: Optional[PyVector3D] = None,
    scale: Optional[PyVector3D] = None,
    inplace: bool = True,
) -> Object:
    if not is_mesh(obj):
        return obj
    if not inplace:
        obj = duplicate(obj)
    focus(obj)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    old_location = deepcopy(obj.location)
    obj.location = Vector((0, 0, 0))
    bpy.ops.object.transform_apply(
        location=True, rotation=True, scale=True
    )  # In case there is unapplied transform
    if scale is not None:
        obj.scale = Vector(scale)
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True
        )
    if rotation is not None:
        obj.rotation_euler = Euler(map(radians, rotation))
        bpy.ops.object.transform_apply(
            location=False, rotation=True, scale=False
        )
    obj.location = Vector(position) if position is not None else old_location
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    return obj


def clamp(value: T, min_value: T, max_value: T) -> T:
    return max(min_value, min(value, max_value))


def normalize_rgb(rgb: ColorRGB) -> tuple[float, float, float]:
    return (
        clamp(rgb[0] / 255, 0.0, 1.0),
        clamp(rgb[1] / 255, 0.0, 1.0),
        clamp(rgb[2] / 255, 0.0, 1.0),
    )


class Builder(object):
    @staticmethod
    def add_material(
        obj: Object,
        *,
        albedo: Optional[str | ColorRGB] = None,
        metallic: Optional[str | float] = None,
        roughness: Optional[str | float] = None,
        specular: Optional[str | float] = None,
        normal: Optional[str] = None,
        height: Optional[tuple[str | float, Midlevel, Scale]] = None,
        emission: Optional[tuple[str | ColorRGB, Strength]] = None,
        ior: float = 1.0,
        backface_culling: bool = False,
        material_name: str = "Material",
        inplace: bool = True,
    ) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)

        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        mat.displacement_method = "BOTH"
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        links.clear()
        bsdf = nodes["Principled BSDF"]
        output = nodes["Material Output"]

        if albedo is not None:
            if isinstance(albedo, str):
                if not exists(albedo):
                    raise FileNotFoundError(f"Texture not found: {albedo}")
                alb = nodes.new("ShaderNodeTexImage")
                alb.image = load_blender_image(albedo)
                links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
            elif isinstance(albedo, tuple):
                if len(albedo) != 3 or not all(
                    isinstance(c, int) for c in albedo
                ):
                    raise ValueError("`albedo` must be a tuple of 3 ints")
                rgba = (*normalize_rgb(albedo), 1)
                bsdf.inputs["Base Color"].default_value = rgba
            else:
                raise TypeError("`albedo` must be either a path or RGB")

        if metallic is not None:
            if isinstance(metallic, str):
                if not exists(metallic):
                    raise FileNotFoundError(f"Texture not found: {metallic}")
                met = nodes.new("ShaderNodeTexImage")
                met.image = load_blender_image(metallic)
                met.image.colorspace_settings.name = "Non-Color"
                links.new(met.outputs["Color"], bsdf.inputs["Metallic"])
            elif isinstance(metallic, (float, int)):
                metallic = clamp(metallic, 0.0, 1.0)
                bsdf.inputs["Metallic"].default_value = metallic
            else:
                raise TypeError("`metallic` must be either a path or float")

        if roughness is not None:
            if isinstance(roughness, str):
                if not exists(roughness):
                    raise FileNotFoundError(f"Texture not found: {roughness}")
                rou = nodes.new("ShaderNodeTexImage")
                rou.image = load_blender_image(roughness)
                rou.image.colorspace_settings.name = "Non-Color"
                links.new(rou.outputs["Color"], bsdf.inputs["Roughness"])
            elif isinstance(roughness, (float, int)):
                roughness = clamp(roughness, 0.0, 1.0)
                bsdf.inputs["Roughness"].default_value = roughness
            else:
                raise TypeError("`roughness` must be either a path or float")

        if specular is not None:
            if isinstance(specular, str):
                if not exists(specular):
                    raise FileNotFoundError(f"Texture not found: {specular}")
                spe = nodes.new("ShaderNodeTexImage")
                spe.image = load_blender_image(specular)
                spe.image.colorspace_settings.name = "Non-Color"
                links.new(
                    spe.outputs["Color"], bsdf.inputs["Specular IOR Level"]
                )
            elif isinstance(specular, (float, int)):
                specular = clamp(specular, 0.0, 1.0)
                bsdf.inputs["Specular IOR Level"].default_value = specular
            else:
                raise TypeError("`specular` must be either a path or float")

        bsdf.inputs["IOR"].default_value = max(ior, 1.0)

        if normal is not None:
            if not exists(normal):
                raise FileNotFoundError(f"Texture not found: {normal}")
            nor = nodes.new("ShaderNodeTexImage")
            nor.image = load_blender_image(normal)
            nor.image.colorspace_settings.name = "Non-Color"
            nmap = nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = 1.0
            links.new(nor.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

        if height is not None:
            if not isinstance(height, tuple) or len(height) != 3:
                raise ValueError("`height` must be (path/float, mid, scale)")
            h1, h2, h3 = height
            disn = nodes.new("ShaderNodeDisplacement")
            if isinstance(h1, str):
                if not exists(h1):
                    raise FileNotFoundError(f"Texture not found: {h1}")
                hei = nodes.new("ShaderNodeTexImage")
                hei.image = load_blender_image(h1)
                hei.image.colorspace_settings.name = "Non-Color"
                links.new(hei.outputs["Color"], disn.inputs["Height"])

                # Add subdivision modifier
                modifier = obj.modifiers.new(
                    name="SubdivisionModifier", type="SUBSURF"
                )
                modifier.subdivision_type = "SIMPLE"
                modifier.render_levels = 6
                modifier.use_adaptive_subdivision = True
                modifier.quality = 6
            elif isinstance(h1, (float, int)):
                disn.inputs["Height"].default_value = max(h1, 0.0)
            else:
                raise TypeError("`height`[0] must be either a path or float")
            disn.inputs["Midlevel"].default_value = max(h2, 0.0)
            disn.inputs["Scale"].default_value = max(h3, 0.0)
            links.new(
                disn.outputs["Displacement"], output.inputs["Displacement"]
            )

        if emission is not None:
            if not isinstance(emission, tuple) or len(emission) != 2:
                raise ValueError("`emission` must be (path/RGB, strength)")
            e1, e2 = emission
            if isinstance(e1, str):
                if not exists(e1):
                    raise FileNotFoundError(f"Texture not found: {e1}")
                emi = nodes.new("ShaderNodeTexImage")
                emi.image = load_blender_image(e1)
                links.new(emi.outputs["Color"], bsdf.inputs["Emission Color"])
            elif isinstance(e1, tuple):
                if len(e1) != 3 or not all(isinstance(c, int) for c in e1):
                    raise ValueError("`emission`[0] must be a tuple of 3 ints")
                rgba = (*normalize_rgb(e1), 1)
                bsdf.inputs["Emission Color"].default_value = rgba
            else:
                raise TypeError("`emission`[0] must be either a path or RGB")
            bsdf.inputs["Emission Strength"].default_value = max(e2, 0.0)

        if not backface_culling:
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        else:
            mix = nodes.new("ShaderNodeMixShader")
            mix.inputs["Fac"].default_value = 1.0
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            geometry = nodes.new("ShaderNodeNewGeometry")
            links.new(bsdf.outputs["BSDF"], mix.inputs[2])
            links.new(mix.outputs["Shader"], output.inputs["Surface"])
            links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
            links.new(transparent.outputs["BSDF"], mix.inputs[1])

        return obj

    @staticmethod
    def bake_vertex_colors_to_albedo(
        obj: Object, albedo_path: str, resolution: int = 1024
    ) -> bool:
        is_mesh(obj, raise_err=True)
        if len(obj.data.materials) == 0:
            raise RuntimeError(
                "`obj` has no material to bake vertex colors to albedo texture"
            )
        if len(obj.data.uv_layers) == 0:
            Builder.create_uv(obj)
        mat = obj.data.materials[0]
        nodes = mat.node_tree.nodes
        img = bpy.data.images.new(
            "BakedAlbedo", width=resolution, height=resolution
        )
        img.file_format = "PNG"
        img_node = nodes.new(type="ShaderNodeTexImage")
        img_node.image = img
        img_node.select = True
        nodes.active = img_node
        bpy.ops.object.bake(
            type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True
        )
        img.filepath_raw = albedo_path
        img.save()
        return exists(albedo_path)

    @staticmethod
    def create_uv(
        obj: Object,
        angle_limit: float = 66.0,
        island_margin: float = 0.0,
        inplace: bool = True,
    ) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=radians(angle_limit), island_margin=island_margin
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        return obj

    @staticmethod
    def create_wireframe(
        obj: Object, sphere_radius: float = 0.05, cylinder_radius: float = 0.02
    ) -> Object:
        is_mesh(obj, raise_err=True)
        obj_copy = duplicate(obj)
        Builder.simplify(obj_copy)
        combined_mesh = bpy.data.meshes.new(f"{obj.name}_WireframeMesh")
        bm = bmesh.new()

        # Create spheres for vertices
        for vert in obj_copy.data.vertices:
            sphere = bmesh.ops.create_icosphere(
                bm, subdivisions=3, radius=sphere_radius
            )
            for v in sphere["verts"]:
                v.co += vert.co  # Move to the vertex location

        # Create cylinders for edges
        for edge in obj_copy.data.edges:
            v1 = obj_copy.data.vertices[edge.vertices[0]].co
            v2 = obj_copy.data.vertices[edge.vertices[1]].co

            # Calculate the midpoint and direction
            mid_point = (v1 + v2) / 2
            direction = (v2 - v1).normalized()
            distance = (v2 - v1).length

            # Create a cylinder
            cylinder = bmesh.ops.create_cone(
                bm,
                cap_tris=True,
                radius1=cylinder_radius,
                radius2=cylinder_radius,
                depth=distance,
                segments=8,
            )

            # Calculate the rotation matrix to align the cylinder
            z_axis = Vector((0, 0, 1))
            rotation_axis = direction.cross(z_axis).normalized()
            rotation_angle = direction.angle(z_axis)
            rotation_matrix = Matrix.Rotation(rotation_angle, 4, rotation_axis)

            # Apply the rotation to the cylinder vertices
            for v in cylinder["verts"]:
                v.co = rotation_matrix @ v.co

            # Move the cylinder to the midpoint
            for v in cylinder["verts"]:
                v.co += mid_point

        # Finalize the bmesh
        bm.to_mesh(combined_mesh)
        bm.free()
        bpy.data.objects.remove(obj_copy, do_unlink=True)
        combined_object = bpy.data.objects.new(
            f"{obj.name}_Wireframe", combined_mesh
        )
        bpy.context.collection.objects.link(combined_object)
        focus(combined_object)
        return combined_object

    @staticmethod
    def flip_normals(obj: Object, inplace: bool = True) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        return obj

    @staticmethod
    def map_faces_to_full_uv(obj: Object, inplace: bool = True) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        uv_layer = obj.data.uv_layers.active.data
        for poly in obj.data.polygons:
            if poly.loop_total == 3 or poly.loop_total == 4:
                if poly.loop_total == 3:
                    uv_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
                else:
                    uv_coords = [
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0),
                    ]
                for i, loop_index in enumerate(poly.loop_indices):
                    uv_layer[loop_index].uv = uv_coords[i % len(uv_coords)]
            else:
                pass
        return obj

    @staticmethod
    def modify_boolean(
        op: Literal["DIFFERENCE", "INTERSECT", "UNION"],
        obj1: Object,
        obj2: Object,
    ) -> Object:
        is_mesh(obj1, raise_err=True)
        is_mesh(obj2, raise_err=True)
        obj = duplicate(obj1, name=f"{obj1.name}_{op}")
        modifier = obj.modifiers.new(name="BooleanModifier", type="BOOLEAN")
        modifier.operation = op
        modifier.object = obj2
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return obj

    @staticmethod
    def new_cube(name: str = "Cube") -> Object:
        bpy.ops.mesh.primitive_cube_add(size=1, calc_uvs=True)
        obj = bpy.context.selected_objects[0]
        obj.name = name
        focus(obj)
        return obj

    @staticmethod
    def new_plane(name: str = "Plane") -> Object:
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.selected_objects[0]
        obj.name = name
        focus(obj)
        return obj

    @staticmethod
    def normalize(
        obj: Object, dims: PyVector3D = (1.0, 1.0, 1.0), inplace: bool = True
    ) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        transform(obj)
        scale_factors = [1 / d if d != 0 else 1.0 for d in obj.dimensions]
        new_scale = (
            obj.scale.x * scale_factors[0] * dims[0],
            obj.scale.y * scale_factors[1] * dims[1],
            obj.scale.z * scale_factors[2] * dims[2],
        )
        transform(obj, scale=new_scale)
        return obj

    @staticmethod
    def separate_bottom(
        obj: Object, bottom_z: float, eps: float = 1e-5
    ) -> tuple[Optional[Object], Object]:
        is_mesh(obj, raise_err=True)
        non_bottom = duplicate(obj, name=f"{obj.name}_non-bottom")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)

        # Get mesh from the object
        bm = bmesh.from_edit_mesh(non_bottom.data)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.select = False

        # Find bottom faces
        has_bottom_face = False
        for face in bm.faces:
            center = face.calc_center_median()
            if center.z < bottom_z + eps:
                face.select = True
                has_bottom_face = True

        # Separate bottom faces from the rest
        bottom = None
        if has_bottom_face:
            bmesh.update_edit_mesh(non_bottom.data)
            ori_objects = set(bpy.context.scene.objects)
            bpy.ops.mesh.separate(type="SELECTED")
            bpy.ops.object.mode_set(mode="OBJECT")
            new_objects = set(bpy.context.scene.objects) - ori_objects
            if len(new_objects) > 0:
                bottom = new_objects.pop()
                bottom.name = f"{obj.name}_bottom"
        else:
            bpy.ops.object.mode_set(mode="OBJECT")

        bm.free()
        return bottom, non_bottom

    @staticmethod
    def separate_top(
        obj: Object, top_z: float, eps: float = 1e-5
    ) -> tuple[Object, Object]:
        is_mesh(obj, raise_err=True)
        non_top = duplicate(obj, name=f"{obj.name}_non-top")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)

        # Get mesh from the object
        bm = bmesh.from_edit_mesh(non_top.data)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.select = False

        # Find top faces
        has_top_face = False
        for face in bm.faces:
            center = face.calc_center_median()
            if center.z > top_z - eps:
                face.select = True
                has_top_face = True

        # Separate top faces from the rest
        top = None
        if has_top_face:
            bmesh.update_edit_mesh(non_top.data)
            ori_objects = set(bpy.context.scene.objects)
            bpy.ops.mesh.separate(type="SELECTED")
            bpy.ops.object.mode_set(mode="OBJECT")
            new_objects = set(bpy.context.scene.objects) - ori_objects
            if len(new_objects) > 0:
                top = new_objects.pop()
                top.name = f"{obj.name}_top"
        else:
            bpy.ops.object.mode_set(mode="OBJECT")

        bm.free()
        return top, non_top

    @staticmethod
    def simplify(obj: Object, inplace: bool = True) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.dissolve_limited()
        bpy.ops.object.mode_set(mode="OBJECT")
        return obj

    @staticmethod
    def solidify(
        obj: Object,
        thickness: float = 0.01,
        offset: float = 1.0,
        inplace: bool = True,
    ) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        modifier = obj.modifiers.new(name="SolidifyModifier", type="SOLIDIFY")
        modifier.thickness = thickness
        modifier.offset = offset
        modifier.use_even_offset = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return obj

    @staticmethod
    def triangulate(obj: Object, inplace: bool = True) -> Object:
        is_mesh(obj, raise_err=True)
        if not inplace:
            obj = duplicate(obj)
        focus(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(obj.data)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bmesh.update_edit_mesh(obj.data)
        bm.free()
        bpy.ops.object.mode_set(mode="OBJECT")
        return obj


class Renderer(object):
    def __init__(self):
        self._world_vertices: list[Vector] = []

    def add_area_light(
        self,
        position: PyVector3D = (0, 0, 0),
        rotation: PyVector3D = (0, 0, 0),
        energy: int = 100,
    ) -> None:
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.selected_objects[0]
        transform(light, position=position, rotation=rotation)
        light.data.energy = energy
        light.data.shape = "SQUARE"
        light.data.size = 1
        light.data.use_shadow = True
        light.data.spread = radians(180)

    def add_point_light(
        self,
        position: Union[Vector, PyVector3D] = (0, 0, 0),
        energy: int = 40,
        use_shadow: bool = True,
    ) -> None:
        bpy.ops.object.light_add(type="POINT")
        light = bpy.context.selected_objects[0]
        light.location = position
        light.data.energy = energy
        light.data.use_soft_falloff = True
        light.data.shadow_soft_size = 0.1
        light.data.use_shadow = use_shadow

    def add_sun_light(
        self,
        position: PyVector3D = (0, 0, 0),
        rotation: PyVector3D = (0, 0, 0),
        energy: float = 5.0,
        angle: float = 0.0,
        use_shadow: bool = True,
    ) -> None:
        bpy.ops.object.light_add(type="SUN")
        light = bpy.context.selected_objects[0]
        transform(light, position=position, rotation=rotation)
        light.data.energy = energy
        light.data.angle = radians(angle)
        light.data.use_shadow = use_shadow

    def compute_bounding_sphere(self) -> tuple[Vector, float]:
        min_x, min_y, min_z = [math.inf for _ in range(3)]
        max_x, max_y, max_z = [-math.inf for _ in range(3)]
        for obj in bpy.context.scene.objects:
            if is_mesh(obj) and obj.visible_get():
                for v in obj.data.vertices:
                    world_v = obj.matrix_world @ v.co
                    min_x = min(min_x, world_v.x)
                    min_y = min(min_y, world_v.y)
                    min_z = min(min_z, world_v.z)
                    max_x = max(max_x, world_v.x)
                    max_y = max(max_y, world_v.y)
                    max_z = max(max_z, world_v.z)
        center = Vector(
            ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        )
        radius = (Vector((max_x, max_y, max_z)) - center).length
        return center, radius

    def compute_world_vertices(self) -> None:
        self._world_vertices = [
            obj.matrix_world @ v.co
            for obj in bpy.context.scene.objects
            if is_mesh(obj) and obj.visible_get()
            for v in obj.data.vertices
        ]

    def disable_shadow(self) -> None:
        for obj in bpy.context.scene.objects:
            if is_mesh(obj):
                obj.visible_shadow = False

    def enable_shadow(self) -> None:
        for obj in bpy.context.scene.objects:
            if is_mesh(obj):
                obj.visible_shadow = True

    def render_panoramic(
        self,
        output_png: str,
        position: PyVector3D,
        *,
        resolution: int = 512,
        background: Optional[tuple[int, int, int]] = None,
    ) -> bool:
        render = bpy.context.scene.render
        render.resolution_x = resolution * 2
        render.resolution_y = resolution

        bpy.ops.object.camera_add()
        camera = bpy.context.selected_objects[0]
        bpy.context.scene.camera = camera
        transform(camera, position=position, rotation=(90, 0, 0))
        camera.data.type = "PANO"
        camera.data.panorama_type = "EQUIRECTANGULAR"
        try:
            bpy.context.scene.render.filepath = output_png
            with redirect_stdout():
                bpy.ops.render.render(write_still=True)
            if background is not None:
                self.add_bg_to_rgba(output_png, output_png, color=background)
        finally:
            bpy.data.objects.remove(camera, do_unlink=True)
        return exists(output_png)

    @overload
    def render_perspective(
        self,
        output_png: str,
        position_or_center: PyVector3D,
        radius: Literal[None] = None,
        *,
        rotation: PyVector3D = ...,
        resolution: Union[int, tuple[int, int]] = ...,
        focal_length: float = ...,
        fit_ratio: float = ...,
        clip_start: float = ...,
        background: Optional[tuple[int, int, int]] = ...,
    ) -> bool: ...

    @overload
    def render_perspective(
        self,
        output_png: str,
        position_or_center: Vector,
        radius: float,
        *,
        rotation: PyVector3D = ...,
        resolution: Union[int, tuple[int, int]] = ...,
        focal_length: float = ...,
        fit_ratio: float = ...,
        clip_start: float = ...,
        background: Optional[tuple[int, int, int]] = ...,
    ) -> bool: ...

    def render_perspective(
        self,
        output_png: str,
        position_or_center: Union[PyVector3D, Vector],
        radius: Optional[float] = None,
        *,
        rotation: PyVector3D = (0, 0, 0),
        resolution: Union[int, tuple[int, int]] = 1024,
        focal_length: float = 50.0,
        fit_ratio: float = 0.0,
        clip_start: float = 0.01,
        background: Optional[tuple[int, int, int]] = None,
    ) -> bool:
        render = bpy.context.scene.render
        render.filepath = output_png
        if isinstance(resolution, tuple):
            render.resolution_x, render.resolution_y = (
                max(resolution[0], 4),
                max(resolution[1], 4),
            )
        else:
            render.resolution_x = render.resolution_y = max(resolution, 4)
        focal_length = clamp(focal_length, 1.0, 5000.0)
        fit_ratio = clamp(fit_ratio, 0.0, 1.0)

        def set_camera(camera: Object) -> None:
            """
            This block of code is responsible for intelligently positioning the camera. It can
            operate in two primary modes, and smoothly interpolate between them:

            1. **Bounding Sphere Method**: A simple, robust method that guarantees the entire
               scene is visible by placing the camera on the edge of a sphere that
               encompasses all objects.

            2. **Tight-Fit Method**: A more complex method that calculates the *exact*
               distance required to make the object's projection perfectly fill the camera's
               viewport, eliminating unnecessary empty space.

            The `fit_ratio` parameter controls the linear interpolation between the camera
            distances calculated by these two methods.

            ---

            ### Coordinate Systems Explained

            To understand the math, it's vital to understand the journey of a vertex through
            different coordinate systems.

            1. **Object (or Local) Space**:
               - This is the coordinate system in which a model is originally designed.
               - The origin (0,0,0) is the object's own pivot point.
               - In the code, `v.co` represents a vertex's position in Object Space.

            2. **World Space**:
               - This is the global coordinate system for the entire Blender scene.
               - All objects, lights, and the camera are positioned and oriented within this
                 single, shared space.
               - We transform from Object to World Space using the object's world matrix.
               - In the code: `world_vert = obj.matrix_world @ v.co`

            3. **Camera View (or Eye) Space**:
               - This is a coordinate system where the camera is conceptually at the origin
                 (0,0,0) and looking down one of its local axes (in Blender, typically the -Z axis).
               - This space is essential because perspective projection calculations become
                 much simpler. We no longer need to worry about the camera's world position
                 or rotation; we just need to know a vertex's position *relative* to the camera.
               - We transform from World to Camera View Space by applying the *inverse* of
                 the camera's rotation matrix.
               - In the code: `cam_space_vec = cam_rot_matrix @ vec_from_center`

            ---

            ### The Mathematical Journey and Derivations

            #### Step 1: Bounding Sphere Distance (`fit_ratio = 0.0`)

            This is the baseline calculation. We imagine a right-angled triangle formed by the
            camera, the center of the bounding sphere, and a tangent of the sphere from the camera.

            ```
            #          Camera
            #            .
            #           /|
            #          / |
            #         /  |
            #        / a | <-- a = half of the FOV = camera.data.angle / 2
            #     d /    |
            #      /     | Tangent
            #     /      |
            #    /       |
            #   /        |
            #  /_________| Right Angle (90°)
            # O     r
            ```

            From trigonometry, we know that `sin(a) = Opposite / Hypotenuse`.
            In our triangle, `Opposite = r` and `Hypotenuse = d`.

            > `sin(a) = r / d`

            Rearranging to solve for the distance `d`:

            > `d = r / sin(a)`

            This is what the code calculates:
            `sphere_dist = radius / sin(camera.data.angle / 2)`

            #### Step 2: Tight-Fit Distance (`fit_ratio = 1.0`)

            This is more complex. The goal is to find the minimum distance `d` such that *every vertex*
            of the object is inside the camera's viewing frustum. We must find the one vertex that
            requires the camera to be furthest away and use that distance.

            **A. From World to Camera View Space:**
            - We first get all vertices in World Space (`all_verts`).
            - We create a rotation matrix that will align any vector with the camera's view.
              `cam_rot_matrix` is the *inverse* of the camera's world rotation. Applying it to a
              vector effectively transforms the vector into the camera's local coordinate system.
            - For each vertex, we find its position relative to the scene's center:
              `vec_from_center = world_vert - position_or_center`.
            - We then rotate this vector into the camera's view:
              `cam_space_vec = cam_rot_matrix @ vec_from_center`.
              The result `(vx, vy, vz)` is the vertex's position in Camera View Space.

            **B. The Core Insight: Calculating the "Perfect Fit" Distance:**
            The fundamental challenge is this: For any given vertex, we need to find the camera
            distance `d` that places that vertex exactly on the edge of the camera's view.
            Imagine a top-down view. The camera is at the origin `C`.
            The object's center `O` is some distance `d` away along the camera's line of sight.
            A vertex `V` is located at `(vx, vz)` *relative to the object's center `O`*.

            ```
            #                    . V
            #                   /|
            #                 /  |
            #               /    |
            #             /      |
            #           /        | abs(vx)
            #         /          |
            #       /            |
            #     /  a           |
            # C /________________|_________ O
            #   |<- - - - - d - - - - - ->|
            #                    |<- vz ->|
            ```

            **The Step-by-Step Derivation**
            1.  **Apply Trigonometry:** We use the tangent function, which relates the angle to the opposite and adjacent sides:
                > `tan(a) = Opposite / Adjacent`

            2.  **Substitute the Known Values:** Now, we plug in the values from our diagram:
                > `tan(a) = abs(vx) / (d - vz)`

            3.  **Solve for `d`:** Our goal is to find `d`. We just need to rearrange the equation algebraically.
                *   Multiply both sides by `(d - vz)`:
                    > `(d - vz) * tan(a) = abs(vx)`
                *   Divide both sides by `tan(a)`:
                    > `d - vz = abs(vx) / tan(a)`
                *   Add `vz` to both sides to isolate `d`:
                    > `d = vz + abs(vx) / tan(a)`

            This final formula is exactly what is used in the code: `dist_x = vz + abs(vx) / tan(fov_x / 2)`.
            We do the same for the vertical axis (`dist_y`) using `fov_y`.

            **C. Finding the Maximum:**
            We must be far enough away to see *all* vertices. Therefore, we loop through every
            vertex, calculate its required `dist_x` and `dist_y`, and the final `fit_dist` is
            the maximum of all these values found.

            #### Step 3: Interpolation

            Finally, we use a standard linear interpolation (lerp) formula to find the final
            camera distance based on the `fit_ratio`.

            > `distance = sphere_dist * (1 - fit_ratio) + fit_dist * fit_ratio`

            - If `fit_ratio` is 0, `distance = sphere_dist`.
            - If `fit_ratio` is 1, `distance = fit_dist`.
            - If `fit_ratio` is 0.5, the distance is exactly halfway between the two.

            #### Step 4: Final Placement

            The camera is positioned by starting at the center of the scene (`position_or_center`)
            and moving *backwards* along the viewing direction by the final calculated `distance`.
            """

            camera.data.type = "PERSP"
            camera.data.lens_unit = "MILLIMETERS"
            camera.data.lens = focal_length
            camera.data.clip_start = clip_start

            if radius is not None:
                # Calculate the base distance using the bounding sphere (corresponds to fit_ratio=0)
                distance = sphere_dist = radius / sin(camera.data.angle / 2)

                if fit_ratio > 0.0:
                    if len(self._world_vertices) == 0:
                        self.compute_world_vertices()
                    if len(self._world_vertices) > 0:
                        aspect_ratio = (
                            render.resolution_x / render.resolution_y
                        )
                        fov_x = camera.data.angle

                        # Calculate the vertical FOV (fov_y) based on the horizontal FOV and aspect ratio.
                        # The formula re-arranges the relationship: tan(fov_x / 2) = aspect_ratio * tan(fov_y / 2)
                        # This ensures that our framing calculations work correctly for non-square images.
                        fov_y = 2 * atan(tan(fov_x / 2) / aspect_ratio)

                        # Create a rotation matrix that will transform world coordinates into camera-relative coordinates.
                        # It works in three steps:
                        # a. Euler(map(radians, rotation)): Takes the camera's human-readable XYZ rotation (in degrees)
                        #    and converts it into a formal Euler rotation object (in radians).
                        # b. .to_matrix(): Converts the Euler rotation into a standard 3x3 rotation matrix. This matrix
                        #    can transform a vector from the camera's local space into world space.
                        # c. .inverted(): We need the opposite. This inverts the matrix, giving us a transform that
                        #    takes a vector from world space and shows us how it looks from the camera's point of view.
                        cam_rot_matrix = (
                            Euler(map(radians, rotation)).to_matrix()
                        ).inverted()

                        fit_dist = 0.0
                        for world_vert in self._world_vertices:
                            # First, calculate the vertex's position relative to the object's own center point.
                            # This gives us a vector in world space.
                            vec_from_center = world_vert - position_or_center

                            # Apply the inverted camera rotation to the vector. This transforms the vertex's
                            # position into "Camera View Space". The resulting coordinates (vx, vy, vz) tell us
                            # where the vertex is as if the camera were at the origin, looking down the -Z axis.
                            cam_space_vec = cam_rot_matrix @ vec_from_center
                            vx, vy, vz = (
                                cam_space_vec  # vx=right, vy=up, vz=depth from object center
                            )

                            # --- Calculate the necessary camera distance for this single vertex ---
                            # This is the core trigonometric calculation. For a given vertex, we find the distance 'd'
                            # the camera needs to be from the object's center to place that vertex exactly on the
                            # edge of the camera's view frustum.
                            # The formula is derived from: tan(FOV/2) = Opposite/Adjacent = abs(v) / (d - vz)
                            # We rearrange it to solve for d: d = vz + abs(v) / tan(FOV/2)
                            dist_x = vz + abs(vx) / tan(fov_x / 2)
                            dist_y = vz + abs(vy) / tan(fov_y / 2)

                            # The true required distance for this vertex is the larger of the two. We then compare
                            # this to the largest distance found so far and update our 'fit_dist' if needed.
                            # By the end of the loop, 'fit_dist' will hold the single distance that guarantees
                            # the entire object fits perfectly within the camera frame.
                            fit_dist = max(fit_dist, dist_x, dist_y)

                        # The 'fit_ratio' slider (from 0.0 to 1.0) determines the final result.
                        # 0.0 = Use only the simple bounding sphere distance (guaranteed to contain the object, but loose).
                        # 1.0 = Use only the precise vertex-calculated distance (a perfect, tight fit).
                        # Values in between allow for a smooth interpolation, giving the user control over the padding.
                        distance = (
                            sphere_dist * (1 - fit_ratio)
                            + fit_dist * fit_ratio
                        )
                    else:
                        warnings.warn(
                            "No visible vertices found in the scene. Using bounding sphere distance only."
                        )

                # Position the camera using the final calculated distance
                direction = Euler(map(radians, rotation)).to_matrix() @ Vector(
                    (0, 0, -1)
                )
                cam_pos = position_or_center - direction * distance
                camera.data.clip_end = distance + 2 * radius * 1.01
            else:
                # Use explicit camera position
                cam_pos = Vector(position_or_center)
                camera.data.clip_end = 1000.0
            camera.location = cam_pos
            camera.rotation_euler = Euler(map(radians, rotation))

        bpy.ops.object.camera_add()
        camera = bpy.context.selected_objects[0]
        bpy.context.scene.camera = camera
        try:
            set_camera(camera)
            with redirect_stdout():
                bpy.ops.render.render(write_still=True)
            if background is not None:
                self.add_bg_to_rgba(output_png, output_png, color=background)
        finally:
            bpy.data.objects.remove(camera, do_unlink=True)
        return exists(output_png)

    @staticmethod
    def add_bg_to_rgba(
        input_path: str, output_path: str, color: ColorRGB = (255, 255, 255)
    ) -> None:
        from PIL import Image

        img = Image.open(input_path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (*color, 255))
        Image.alpha_composite(bg, img).convert("RGB").save(output_path)

    @staticmethod
    def rotate_image(
        input_path: str, output_path: str, angle: DegreeCCW
    ) -> None:
        from PIL import Image

        img = Image.open(input_path)
        img.rotate(angle, expand=True).save(output_path)
