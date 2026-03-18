from __future__ import annotations

import io
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from logger_config import logger
from libs.trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor


class DINOScorer:
    """
    Deterministic candidate scorer using DINO v3 patch-level feature matching.

    Compares each candidate's rendered views against the input prompt image
    using patch-level cosine similarity. Unlike VLLM-based judging, this is
    reproducible across different GPU hardware because cosine similarity
    rankings are robust to small floating-point perturbations.
    """

    def __init__(self, dino_model: DinoV3FeatureExtractor):
        self.dino_model = dino_model

    @torch.no_grad()
    def score_candidates(
        self,
        prompt_image: Image.Image,
        candidates_views: List[List[Image.Image]],
    ) -> Tuple[int, List[float]]:
        """
        Score candidates using patch-level DINO feature matching.

        For each candidate:
        1. Extract DINO patch features from each of its rendered views
        2. Extract DINO patch features from the input prompt image
        3. Compute patch-level cosine similarity (each input patch matched
           to best candidate patch across all views)
        4. Average match quality across input patches

        Args:
            prompt_image: The original input image (PIL Image).
            candidates_views: List of [num_candidates][num_views] PIL Images.

        Returns:
            Tuple of (best_index, scores_list).
        """
        prompt_features = self.dino_model([prompt_image])
        prompt_features = F.normalize(prompt_features[0], dim=-1)  # [N_prompt, D]

        scores = []
        for views in candidates_views:
            view_features = self.dino_model(views)  # [num_views, N_view, D]
            view_features = F.normalize(view_features, dim=-1)

            view_scores = []
            for v in range(view_features.shape[0]):
                # Similarity matrix: [N_prompt, N_view]
                sim_matrix = torch.mm(prompt_features, view_features[v].t())
                # Best match for each prompt patch, then average
                best_matches = sim_matrix.max(dim=1).values  # [N_prompt]
                view_scores.append(best_matches.mean().item())

            scores.append(sum(view_scores) / len(view_scores))

        # Stability margin: only prefer a candidate over index 0 if its
        # score is meaningfully higher, not just within FP noise.
        SCORE_MARGIN = 0.001
        best_idx = 0
        best_score = scores[0]
        for i, score in enumerate(scores[1:], 1):
            if score > best_score + SCORE_MARGIN:
                best_idx = i
                best_score = score
        logger.info(
            f"DINO scores: {[f'{s:.4f}' for s in scores]} | "
            f"Best: candidate {best_idx} ({scores[best_idx]:.4f})"
        )
        return best_idx, scores

    @torch.no_grad()
    def score_images(
        self,
        reference_image: Image.Image,
        candidate_images: List[Image.Image],
    ) -> Tuple[int, List[float]]:
        """
        Score individual candidate images against a reference image using
        patch-level DINO feature matching.

        Args:
            reference_image: The original input image to compare against.
            candidate_images: List of candidate images to score.

        Returns:
            Tuple of (best_index, scores_list).
        """
        ref_features = self.dino_model([reference_image])
        ref_features = F.normalize(ref_features[0], dim=-1)  # [N_ref, D]

        scores = []
        for img in candidate_images:
            img_features = self.dino_model([img])
            img_features = F.normalize(img_features[0], dim=-1)  # [N_img, D]
            sim_matrix = torch.mm(ref_features, img_features.t())
            best_matches = sim_matrix.max(dim=1).values
            scores.append(best_matches.mean().item())

        # Stability margin: only prefer a candidate over index 0 if its
        # score is meaningfully higher, not just within FP noise.
        SCORE_MARGIN = 0.001
        best_idx = 0
        best_score = scores[0]
        for i, score in enumerate(scores[1:], 1):
            if score > best_score + SCORE_MARGIN:
                best_idx = i
                best_score = score
        logger.info(
            f"DINO image scores: {[f'{s:.4f}' for s in scores]} | "
            f"Best: candidate {best_idx} ({scores[best_idx]:.4f})"
        )
        return best_idx, scores
