"""Validate the single-image Pixal3D pipeline and mesh settings at the API boundary."""

from dataclasses import dataclass
from math import pi, radians

from scripts.mesh_options import DEFAULT_GENERATION_TRIANGLES, glb_options, texture_size

SAMPLERS = ("sparse_structure_sampler_params", "shape_slat_sampler_params", "tex_slat_sampler_params")
SAMPLER_NUMBERS = {
    "guidance_strength": (0, 20),
    "guidance_rescale": (0, 1),
    "rescale_t": (0.01, 20),
}
CASCADE_TYPES = {1024: "1024_cascade", 1536: "1536_cascade"}


def _number(value, field: str, minimum: float, maximum: float, *, integer: bool = False):
    valid_types = (int,) if integer else (int, float)
    if type(value) not in valid_types or not minimum <= value <= maximum:
        kind = "an integer" if integer else "a number"
        raise ValueError(f"{field} must be {kind} between {minimum} and {maximum}.")
    return value


def _boolean(inputs: dict, name: str, default: bool) -> bool:
    value = inputs.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"input.{name} must be a boolean.")
    return value


def _sampler_options(values, field: str) -> dict:
    if not isinstance(values, dict):
        raise ValueError(f"{field} must be an object.")
    validated = {}
    for name, value in values.items():
        label = f"{field}.{name}"
        if name == "steps":
            value = _number(value, label, 1, 50, integer=True)
        elif name in SAMPLER_NUMBERS:
            value = _number(value, label, *SAMPLER_NUMBERS[name])
        elif name == "guidance_interval":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"{label} must contain two numbers.")
            start, end = (_number(x, label, 0, 1) for x in value)
            if start > end:
                raise ValueError(f"{label} must be in ascending order.")
            value = (start, end)
        else:
            raise ValueError(f"Unsupported sampler parameter: {label}.")
        validated[name] = value
    return validated


def _camera_options(inputs: dict) -> dict:
    scale = _number(inputs.get("mesh_scale", 1.0), "input.mesh_scale", 0.01, 100)
    camera = inputs.get("camera_params")
    if camera is not None:
        if not isinstance(camera, dict) or set(camera) - {"camera_angle_x", "distance", "mesh_scale"}:
            raise ValueError("input.camera_params accepts camera_angle_x, distance and mesh_scale.")
        if any(field in inputs for field in ("manual_fov", "fov_unit", "extend_pixel", "image_resolution")):
            raise ValueError("Use either input.camera_params or FOV and framing options.")
        return {"camera_params": {
            "camera_angle_x": _number(camera.get("camera_angle_x"), "input.camera_params.camera_angle_x", radians(1), radians(170)),
            "distance": _number(camera.get("distance"), "input.camera_params.distance", 0.001, 10_000),
            "mesh_scale": _number(camera.get("mesh_scale", scale), "input.camera_params.mesh_scale", 0.01, 100),
        }}

    fov_unit = inputs.get("fov_unit", "rad")
    if fov_unit not in ("rad", "deg"):
        raise ValueError("input.fov_unit must be rad or deg.")
    manual_fov = inputs.get("manual_fov", -1)
    if manual_fov != -1:
        minimum, maximum = (1, 170) if fov_unit == "deg" else (radians(1), radians(170))
        manual_fov = _number(manual_fov, "input.manual_fov", minimum, maximum)
        if fov_unit == "deg":
            manual_fov = radians(manual_fov)
    return {
        "manual_fov": None if manual_fov == -1 else manual_fov,
        "mesh_scale": scale,
        "extend_pixel": _number(inputs.get("extend_pixel", 0), "input.extend_pixel", 0, 256, integer=True),
        "image_resolution": _number(inputs.get("image_resolution", 512), "input.image_resolution", 128, 2048, integer=True),
    }


@dataclass(frozen=True)
class GenerationOptions:
    pipeline: dict
    camera: dict
    postprocess: dict
    texture_format: str
    low_vram: bool
    num_samples: int


def generation_options(inputs: dict, *, asset_format: str) -> GenerationOptions:
    resolution = inputs.get("resolution")
    pipeline_type = inputs.get("pipeline_type", "1024_cascade")
    if pipeline_type not in CASCADE_TYPES.values():
        raise ValueError("input.pipeline_type must be 1024_cascade or 1536_cascade.")
    if "resolution" in inputs:
        if type(resolution) is not int or resolution not in CASCADE_TYPES:
            raise ValueError("input.resolution must be 1024 or 1536.")
        if "pipeline_type" in inputs and pipeline_type != CASCADE_TYPES[resolution]:
            raise ValueError("input.resolution and input.pipeline_type must specify the same cascade.")
        pipeline_type = CASCADE_TYPES[resolution]
    pipeline = {
        "pipeline_type": pipeline_type,
        # Upstream uses this as a fallback threshold while above 1024.
        "max_num_tokens": _number(inputs.get("max_num_tokens", 49152), "input.max_num_tokens", 1, 65536, integer=True),
        **{name: _sampler_options(inputs.get(name, {}), f"input.{name}") for name in SAMPLERS},
    }
    texture_format = inputs.get("texture_format", "webp" if asset_format == "glb" else "png")
    if asset_format == "glb":
        if texture_format not in ("webp", "png"):
            raise ValueError("input.texture_format must be webp or png for GLB.")
        target, size = glb_options(inputs, default_triangles=DEFAULT_GENERATION_TRIANGLES, max_triangles=1_000_000)
        remesh = _boolean(inputs, "remesh", True)
        if not remesh and ("remesh_band" in inputs or "remesh_project" in inputs):
            raise ValueError("input.remesh_band and input.remesh_project require remesh=true.")
        postprocess = {
            "decimation_target": target,
            "texture_size": size,
            "remesh": remesh,
            "remesh_band": _number(inputs.get("remesh_band", 1), "input.remesh_band", 0.25, 4),
            "remesh_project": _number(inputs.get("remesh_project", 0), "input.remesh_project", 0, 1),
        }
    else:
        if texture_format != "png" or any(name in inputs for name in ("target_triangles", "remesh", "remesh_band", "remesh_project")):
            raise ValueError("USDZ uses PNG and retains the source quads. GLB is required for remeshing and triangle targets.")
        postprocess = {"texture_size": texture_size(inputs), "remesh": True}

    uv_defaults = (
        ("mesh_cluster_threshold_cone_half_angle_rad", pi / 2, 0.01, pi, False),
        ("mesh_cluster_refine_iterations", 0, 0, 8, True),
        ("mesh_cluster_global_iterations", 1, 1, 8, True),
        ("mesh_cluster_smooth_strength", 1, 0, 10, False),
    )
    postprocess.update({
        name: _number(inputs.get(name, default), f"input.{name}", lo, hi, integer=integer)
        for name, default, lo, hi, integer in uv_defaults
    })
    postprocess["verbose"] = _boolean(inputs, "postprocess_verbose", False)
    return GenerationOptions(
        pipeline, _camera_options(inputs), postprocess, texture_format,
        _boolean(inputs, "low_vram", False),
        _number(inputs.get("num_samples", 1), "input.num_samples", 1, 4, integer=True),
    )
