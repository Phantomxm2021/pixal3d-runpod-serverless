"""Run with Blender in background mode to export one static USDZ asset to GLB."""

from pathlib import Path
import sys

import bpy


def main() -> None:
    source, destination, triangle_limit, texture_limit = sys.argv[sys.argv.index("--") + 1:]
    triangle_limit, texture_limit = int(triangle_limit), int(texture_limit)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.ops.wm.usd_import(filepath=source, import_materials=True, import_usd_preview=True) != {"FINISHED"}:
        raise RuntimeError("Could not read source USDZ.")

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError("Expected a single source mesh.")
    mesh = meshes[0]
    mesh.data.validate(clean_customdata=False)
    mesh.data.calc_loop_triangles()
    if not mesh.data.uv_layers or not mesh.data.materials:
        raise RuntimeError("Source mesh is missing UVs or a material.")
    triangle_count = len(mesh.data.loop_triangles)
    if triangle_count > triangle_limit:
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        modifier = mesh.modifiers.new("Triangle budget", "DECIMATE")
        modifier.ratio = triangle_limit / triangle_count
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        mesh.data.validate(clean_customdata=False)

    for image in bpy.data.images:
        width, height = image.size
        largest = max(width, height)
        if largest > texture_limit:
            image.scale(max(1, round(width * texture_limit / largest)),
                        max(1, round(height * texture_limit / largest)))
            image.update()
            image.pack()

    if bpy.ops.export_scene.gltf(
        filepath=destination, export_format="GLB", export_image_format="AUTO",
    ) != {"FINISHED"} or not Path(destination).is_file():
        raise RuntimeError("Blender did not write a GLB.")


if __name__ == "__main__":
    main()
