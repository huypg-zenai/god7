from pydantic import BaseModel
from .enums import TrellisPipeType, TrellisMode


class TrellisConfig(BaseModel):
    """TRELLIS.2 model configuration"""
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    shape_slat_steps: int = 12
    shape_slat_cfg_strength: float = 3.0
    tex_slat_steps: int = 12
    tex_slat_cfg_strength: float = 3.0
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024  # '512', '1024', '1024_cascade', '1536_cascade'
    max_num_tokens: int = 49152
    mode: TrellisMode = TrellisMode.MULTIDIFFUSION
    multiview: bool = False
    num_candidates: int = 2
    num_edit_candidates: int = 3
    voxel_complexity_thresholds: dict[int, int] = {3: 20000, 2: 30000, 1: 40000}
    offload_decoders: bool = True
    gpu: int = 0
