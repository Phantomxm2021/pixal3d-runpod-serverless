"""Read the geometry counts from the GLB that is actually delivered."""

import json
import struct
from pathlib import Path


def glb_stats(path: Path) -> dict[str, int]:
    size = path.stat().st_size
    with path.open("rb") as file:
        header = file.read(20)
        if len(header) != 20:
            raise ValueError("GLB header is incomplete.")
        magic, version, length, json_size, chunk = struct.unpack("<4sIII4s", header)
        if (magic, version, length, chunk) != (b"glTF", 2, size, b"JSON") or json_size > size - 20:
            raise ValueError("GLB header is invalid.")
        document = json.loads(file.read(json_size))

    accessors = document["accessors"]
    triangles = vertices = 0
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise ValueError("Expected triangle primitives in the exported GLB.")
            position = primitive["attributes"]["POSITION"]
            vertices += accessors[position]["count"]
            index_count = accessors[primitive.get("indices", position)]["count"]
            if index_count % 3:
                raise ValueError("Invalid triangle index count in the exported GLB.")
            triangles += index_count // 3
    if not vertices or not triangles:
        raise ValueError("The exported GLB contains no mesh.")
    return {"triangle_count": triangles, "vertex_count": vertices, "size_bytes": size}
