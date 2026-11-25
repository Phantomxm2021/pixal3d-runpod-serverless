"""Adapt Sprint3D's Torch atlas to the O-Voxel texture baker."""

import torch


def unwrap_gpu(vertices, faces, texture_size: int):
    from sprint3d_uv import AtlasConfig, SegmentConfig, build_atlas, face_normals_areas, segment_charts

    points = vertices.float().contiguous()
    faces = faces.long().contiguous()
    if not len(points) or not len(faces):
        raise ValueError("UV unwrapping requires a nonempty mesh.")
    low, high = torch.aminmax(faces)
    if low.item() < 0 or high.item() >= len(points):
        raise ValueError("Mesh contains out-of-range face indices.")
    if not torch.isfinite(points).all().item():
        raise ValueError("Mesh contains non-finite vertex positions.")
    _, area = face_normals_areas(points, faces)
    valid = area > 0
    if not valid.any().item():
        raise ValueError("UV unwrapping requires a mesh with positive surface area.")
    # The GLB decimator can leave zero-area triangles.
    faces = faces[valid]
    stats = {}
    labels = segment_charts(points, faces, SegmentConfig(), stats)
    result = build_atlas(points, faces, labels, texture_size, AtlasConfig(), stats)
    mapping = result["out_vid"]
    print(f"[uv] Sprint3D atlas: {len(faces)} faces, {stats['charts']} charts, "
          f"{stats['overlap_texels']} overlapping texels", flush=True)
    return (
        points[mapping].contiguous(),
        result["out_faces"].contiguous(),
        result["uv"].contiguous(),
        mapping.contiguous(),
    )
