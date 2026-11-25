"""Validate the face and texture budgets for GLB output."""

DEFAULT_GENERATION_TRIANGLES = 200_000
DEFAULT_TEXTURE_SIZE = 2048


def glb_options(inputs: dict, *, default_triangles: int, max_triangles: int = 200_000) -> tuple[int, int]:
    return triangle_budget(inputs, default_triangles=default_triangles, max_triangles=max_triangles), texture_size(inputs)


def triangle_budget(inputs: dict, *, default_triangles: int, max_triangles: int) -> int:
    target_triangles = inputs.get("target_triangles", default_triangles)
    if type(target_triangles) is not int or not 1_000 <= target_triangles <= max_triangles:
        raise ValueError(f"input.target_triangles must be an integer between 1000 and {max_triangles}.")
    return target_triangles


def texture_size(inputs: dict) -> int:
    size = inputs.get("texture_size", DEFAULT_TEXTURE_SIZE)
    if type(size) is not int or size not in (512, 1024, 2048, 4096):
        raise ValueError("input.texture_size must be 512, 1024, 2048 or 4096.")
    return size
