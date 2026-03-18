import base64
import asyncio
import json
import re
from typing import Tuple

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from logger_config import logger
from .settings import JudgeConfig


SYSTEM_PROMPT = """
You are a specialized 3D model evaluation system.
Analyze visual quality and prompt adherence with expert precision.
Always respond with valid JSON only."""

USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""


class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str


class DuelManager:
    def __init__(self, settings: JudgeConfig):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.vllm_url,
            api_key=settings.vllm_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            ),
        )

    async def _call_vllm(
        self, prompt_b64: str, img1_b64: str, img2_b64: str, seed: int
    ) -> JudgeResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{prompt_b64}"},
                    },
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img1_b64}"},
                    },
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img2_b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                response_format=response_format,
                seed=seed,
            )

            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning("vLLM response was truncated due to max_tokens limit")

            content = choice.message.content
            if not content:
                raise ValueError("Empty response from vLLM")
            content = content.strip()
            logger.info(f"vLLM judge response (seed={seed}): {content}")

            # Parse penalties via regex (robust to truncated JSON)
            penalty_1_match = re.search(r'"penalty_1":\s*(\d+)', content)
            penalty_2_match = re.search(r'"penalty_2":\s*(\d+)', content)
            penalty_1 = int(penalty_1_match.group(1)) if penalty_1_match else 5
            penalty_2 = int(penalty_2_match.group(1)) if penalty_2_match else 5

            issues = ""
            if finish_reason != "length":
                try:
                    issues = json.loads(content)["issues"]
                except (json.JSONDecodeError, KeyError):
                    issues = "Incomplete JSON"

            return JudgeResponse(
                penalty_1=penalty_1, penalty_2=penalty_2, issues=issues
            )

        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")

    async def run_duel(
        self, prompt_bytes: bytes, img1_bytes: bytes, img2_bytes: bytes, seed: int
    ) -> Tuple[int, str]:
        """
        Run a position-balanced duel between two candidate images.

        Args:
            prompt_bytes: Original prompt image as PNG bytes
            img1_bytes: First candidate rendered grid as PNG bytes
            img2_bytes: Second candidate rendered grid as PNG bytes
            seed: Random seed for reproducibility

        Returns:
            Tuple of (winner_idx, issues):
                winner_idx: -1 if img1 wins, 1 if img2 wins
                issues: Human-readable issue summary
        """
        if not img1_bytes or not img2_bytes:
            logger.error("Invalid image bytes provided to judge")
            return -1, "Invalid input — defaulting to first candidate"

        prompt_b64 = base64.b64encode(prompt_bytes).decode("utf-8")
        render1_b64 = base64.b64encode(img1_bytes).decode("utf-8")
        render2_b64 = base64.b64encode(img2_bytes).decode("utf-8")

        # Position-balanced: run twice with swapped positions
        logger.info("Running position-balanced VLLM duel...")
        res_direct, res_swapped = await asyncio.gather(
            self._call_vllm(prompt_b64, render1_b64, render2_b64, seed),
            self._call_vllm(prompt_b64, render2_b64, render1_b64, seed),
        )

        score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
        score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
        issues = f"Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"

        logger.info(
            f"Duel scores — Candidate 1: {score1:.1f} (direct={res_direct.penalty_1}, swapped={res_swapped.penalty_2}) | "
            f"Candidate 2: {score2:.1f} (direct={res_direct.penalty_2}, swapped={res_swapped.penalty_1})"
        )

        margin = self.settings.duel_margin
        if score2 - score1 >= margin:
            return -1, issues  # Image 1 clearly better
        elif score1 - score2 >= margin:
            return 1, issues  # Image 2 clearly better
        else:
            return -1, issues  # Too close — default to Image 1 (incumbent)
