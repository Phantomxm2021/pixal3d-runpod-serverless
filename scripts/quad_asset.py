"""Preserve native dual-grid quads and package the baked source as USDZ."""

from pathlib import Path
import tempfile

import numpy as np

QUAD_TRIANGLES = ((0, 1, 2), (0, 2, 3))


def triangle_keys(faces):
    ordered = np.ascontiguousarray(np.sort(faces, axis=1), dtype=np.int32)
    return ordered.view([("a", "i4"), ("b", "i4"), ("c", "i4")]).reshape(-1)


def quad_corner_uvs(quads, atlas_faces, atlas_uvs, source_indices, texture_size):
    """Reassemble triangle UVs into quad corners, keeping each diagonal continuous."""
    expected = quads[:, QUAD_TRIANGLES].reshape(-1, 3)
    actual = source_indices[atlas_faces]
    if not len(actual):
        raise ValueError("UV parameterization returned no faces.")
    actual_keys, expected_keys = triangle_keys(actual), triangle_keys(expected)
    order = np.argsort(actual_keys)
    indices = np.searchsorted(actual_keys, expected_keys, sorter=order)
    matched = order[np.minimum(indices, len(order) - 1)]
    valid = (indices < len(order)) & (actual_keys[matched] == expected_keys)

    corner = (actual[matched, None, :] == expected[:, :, None]).argmax(axis=2)
    uv_ids = np.take_along_axis(atlas_faces[matched], corner, axis=1)
    parts = atlas_uvs[uv_ids].reshape(-1, 2, 3, 2)
    first, second = parts[:, 0], parts[:, 1]
    regular = valid.reshape(-1, 2).all(axis=1)
    regular &= (first[:, (0, 2)] == second[:, (0, 1)]).all(axis=(1, 2))
    uv = np.concatenate((first, second[:, 2:3]), axis=1)
    split = np.flatnonzero(~regular)
    if len(split) == 0:
        return uv

    # Reserve atlas space for quads whose diagonal was cut by the triangle
    # unwrapper. These quads receive individual islands before PBR baking.
    reserve = min(0.95 if regular.any() else 1.0, max(
        0.08, len(split) / len(quads), len(split) * 40 / texture_size ** 2,
    ))
    first_pixel = round(texture_size * (1 - reserve))
    columns = max(1, round(np.sqrt(len(split) * reserve)))
    rows = (len(split) + columns - 1) // columns
    tile_w, tile_h = (texture_size - first_pixel) / columns, texture_size / rows
    if min(tile_w, tile_h) < 5:
        raise ValueError("The quad atlas needs more room for its UV islands.")
    uv[regular, :, 0] *= first_pixel / texture_size
    tiles = np.arange(len(split))
    origins = np.column_stack((
        first_pixel + (tiles % columns) * tile_w, (tiles // columns) * tile_h,
    ))
    offsets = np.asarray([
        (2, 2), (tile_w - 2, 2), (tile_w - 2, tile_h - 2), (2, tile_h - 2),
    ])
    uv[split] = (origins[:, None, :] + offsets) / texture_size
    return uv


class QuadAsset:
    def __init__(self, points, faces, quads):
        if not len(quads) or len(faces) != 2 * len(quads):
            raise ValueError("The decoder did not produce paired quad faces.")
        self.points = points
        # Rotate the four corners when the decoder chose the 1–3 diagonal.
        # The canonical 0–2 split then describes the same surface.
        quads = quads.cpu().numpy()
        other_diagonal = faces[::2, 2].cpu().numpy() == quads[:, 3]
        self.quads = np.where(other_diagonal[:, None], np.roll(quads, -1, axis=1), quads)
        self.faces = faces.new_tensor(self.quads[:, QUAD_TRIANGLES].reshape(-1, 3))

    def triangulated_mesh(self):
        return self.points, self.faces

    def atlas(self, faces, uv, vertex_map, texture_size):
        corners = quad_corner_uvs(
            self.quads, faces.cpu().numpy(), uv.cpu().numpy(), vertex_map.cpu().numpy(), texture_size
        )
        values = np.column_stack([self.quads.reshape(-1), corners.reshape(-1, 2)])
        unique, inverse = np.unique(values, axis=0, return_inverse=True)
        self.uv_quads = inverse.reshape(-1, 4)
        maps = vertex_map.new_tensor(unique[:, 0].astype(np.int64))
        triangles = self.uv_quads[:, QUAD_TRIANGLES].reshape(-1, 3)
        return self.points[maps.long()], faces.new_tensor(triangles), uv.new_tensor(unique[:, 1:]), maps

    def export(self, textured_mesh, output: Path) -> dict[str, int]:
        from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

        original, first, inverse = np.unique(
            self.quads.reshape(-1), return_index=True, return_inverse=True
        )
        vertex_ids = self.uv_quads.reshape(-1)[first]
        positions = np.asarray(textured_mesh.vertices[vertex_ids], dtype=np.float32)
        corner_uv = np.asarray(textured_mesh.visual.uv[self.uv_quads.reshape(-1)], dtype=np.float32)
        material = textured_mesh.visual.material

        with tempfile.TemporaryDirectory(prefix="pixal3d-usd-") as directory:
            directory = Path(directory)
            material.baseColorTexture.save(directory / "color.png")
            material.metallicRoughnessTexture.save(directory / "metal-rough.png")
            source = directory / "asset.usdc"
            stage = Usd.Stage.CreateNew(str(source))
            root = UsdGeom.Xform.Define(stage, "/Asset")
            stage.SetDefaultPrim(root.GetPrim())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(stage, 1)

            mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(self.quads), 4, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inverse.astype(np.int32)))
            normals = np.asarray(textured_mesh.vertex_normals[vertex_ids], dtype=np.float32)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            types = Sdf.ValueTypeNames
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", types.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            primvar.Set(Vt.Vec2fArray.FromNumpy(corner_uv))

            look = UsdShade.Material.Define(stage, "/Asset/Material")
            surface = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
            surface.CreateIdAttr("UsdPreviewSurface")
            look.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", types.Token))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
            reader = UsdShade.Shader.Define(stage, "/Asset/Material/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", types.Token).Set("st")
            st = reader.CreateOutput("result", types.Float2)
            for name, file, color_space, outputs in (
                ("Color", "color.png", "sRGB", (("rgb", "diffuseColor", types.Float3, types.Color3f),)),
                ("MetalRough", "metal-rough.png", "raw", (
                    ("b", "metallic", types.Float, types.Float), ("g", "roughness", types.Float, types.Float),
                )),
            ):
                texture = UsdShade.Shader.Define(stage, f"/Asset/Material/{name}")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", types.Asset).Set(Sdf.AssetPath(f"./{file}"))
                texture.CreateInput("sourceColorSpace", types.Token).Set(color_space)
                texture.CreateInput("st", types.Float2).ConnectToSource(st)
                for channel, parameter, output_type, input_type in outputs:
                    surface.CreateInput(parameter, input_type).ConnectToSource(texture.CreateOutput(channel, output_type))

            stage.GetRootLayer().Save()
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(output)):
                raise RuntimeError("Could not package mesh and textures as USDZ.")

        return {"quad_count": len(self.quads), "vertex_count": len(original)}
