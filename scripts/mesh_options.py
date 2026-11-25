"""Validate the face and texture budgets for GLB output."""

DEFAULT_GENERATION_TRIANGLES = 200_000
DEFAULT_TEXTURE_SIZE = 2048


def glb_options(inputs: dict, *, default_triangles: int) -> tuple[int, int]:
    target_triangles = inputs.get("target_triangles", default_triangles)
    texture_size = inputs.get("texture_size", DEFAULT_TEXTURE_SIZE)
    if type(target_triangles) is not int or not 1_000 <= target_triangles <= 200_000:
        raise ValueError("input.target_triangles must be an integer between 1000 and 200000.")
    if type(texture_size) is not int or texture_size not in (512, 1024, 2048):
        raise ValueError("input.texture_size must be 512, 1024 or 2048.")
    return target_triangles, texture_size
