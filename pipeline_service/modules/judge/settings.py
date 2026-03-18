from pydantic import BaseModel


class JudgeConfig(BaseModel):
    """VLLM Judge configuration"""
    vllm_url: str = "http://localhost:8095/v1"
    vllm_api_key: str = "local"
    vllm_model_name: str = "THUDM/GLM-4.1V-9B-Thinking"
    enabled: bool = True
    duel_margin: float = 1.5
