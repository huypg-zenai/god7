from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional
from urllib import response

from PIL import Image
from modules.converters.params import GLBConverterParams
import torch
import gc

from config.settings import SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.mesh_generator.schemas import TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.ben2_module import BEN2BackgroundRemovalService
from modules.background_removal.birefnet_module import BirefNetBackgroundRemovalService
from modules.grid_renderer.render import GridViewRenderer
from modules.mesh_generator.trellis_manager import TrellisService
from modules.converters.glb_converter import GLBConverter
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.judge.duel_manager import DuelManager
from modules.judge.dino_scorer import DINOScorer
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files
    
class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf, renderer: Optional[GridViewRenderer] = None) -> None:
        self.settings = settings
        self.renderer = renderer

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings.qwen, settings.model_versions)

        # Initialize background removal module
        if self.settings.background_removal.model_id == "PramaLLC/BEN2":
            self.rmbg = BEN2BackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet_dynamic":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

        # Initialize prompting libraries for both modes
        self.prompting_library = PromptingLibrary.from_file(settings.qwen.prompt_path_base)

        # Initialize Trellis module
        self.trellis = TrellisService(settings.trellis, settings.model_versions)
        self.glb_converter = GLBConverter(settings.glb_converter)

        # Initialize VLLM judge
        self.duel_manager = DuelManager(settings.judge) if settings.judge.enabled else None
        # DINO scorer initialized after trellis startup (needs the loaded DINO model)
        self.dino_scorer: Optional[DINOScorer] = None
        
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()

        # Initialize DINO scorer using Trellis's already-loaded DINO model
        self.dino_scorer = DINOScorer(self.trellis.pipeline.image_cond_model)
        logger.info("DINO scorer initialized")

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(512,512),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(temp_image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=42
        )

        result = await self.generate(request)
        
        if result.glb_file_base64 and self.renderer:
            grid_view_bytes = self.renderer.grid_from_glb_bytes(result.glb_file_base64)
            if not grid_view_bytes:
                logger.warning("Grid view generation failed during warmup")

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return GLB as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            output_type: Desired output type (MESH) (default: MESH)
            
        Returns:
            GLB file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )

        response = await self.generate(request)
        
        return response.glb_file_base64 # bytes
    
    def _get_dynamic_glb_params(self, mesh: MeshWithVoxel, request_params, elapsed_time: float):
        """
        Intelligent GLB parameter selection based on remaining time budget.
        Fast tasks get higher quality processing (texture, decimation).
        Slow tasks get reduced params to stay within timeout.
        """
        TIME_TARGET = 78  
        remaining = TIME_TARGET - elapsed_time
        face_count = mesh.faces.shape[0]

        if remaining > 55:
            dynamic = GLBConverterParams.Overrides(
                texture_size=3072,
                decimation_target=400000  
            )
            logger.debug(f"Dynamic GLB: FAST ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=3072, decim=400k")
        elif remaining > 40:
            dynamic = GLBConverterParams.Overrides(
                texture_size=2560,
                decimation_target=300000
            )
            logger.debug(f"Dynamic GLB: NORMAL ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=2560, decim=300k")
        elif remaining > 25:
            logger.debug(f"Dynamic GLB: MODERATE ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> defaults")
            return request_params
        else:
            dynamic = GLBConverterParams.Overrides(
                texture_size=1536,
                decimation_target=180000
            )
            logger.debug(f"Dynamic GLB: SLOW ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=1536, decim=180k")

        if request_params:
            merged = dynamic.model_dump(exclude_none=True)
            merged.update(request_params.model_dump(exclude_none=True))
            return GLBConverterParams.Overrides(**merged)
        
        return dynamic
    
        
    def _edit_images(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Edit image based on current mode (multiview or base).
        
        Args:
            image: Input image to edit
            seed: Random seed for reproducibility
            
        Returns:
            List of edited images
        """


        if self.settings.trellis.multiview:
            logger.info("Multiview mode: generating multiple views")
            views_prompt = self.prompting_library.promptings['views']

            edited_images = []
            for prompt_text in views_prompt.prompt:
                logger.debug(f"Editing view with prompt: {prompt_text}")
                result = self.qwen_edit.edit_image(
                    prompt_image=image,
                    seed=seed,
                    prompting=prompt_text
                )
                edited_images.extend(result)
                
            edited_images.append(image.copy()) # Original image

            return edited_images
        
        # Base mode: generate multiple edit candidates with different seeds
        num_edit_candidates = self.settings.trellis.num_edit_candidates
        base_prompt = self.prompting_library.promptings['base']
        logger.info(f"Base mode: generating {num_edit_candidates} edit candidate(s) with seeds {seed}..{seed + num_edit_candidates - 1}")

        all_edited = []
        for i in range(num_edit_candidates):
            candidate_seed = seed + i
            logger.debug(f"Editing candidate {i+1}/{num_edit_candidates} with seed {candidate_seed}")
            result = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=candidate_seed,
                prompting=base_prompt
            )
            all_edited.extend(result)

        return all_edited

    async def generate_meshes(self, request: GenerationRequest) -> tuple[list[MeshWithVoxel], list[Image.Image], list[Image.Image]]:
        """
        Generate meshes (batch) from Trellis pipeline, along with processed images.
        Uses two-phase generation: shape first, then texture with adaptive candidate count
        based on voxel complexity to prevent OOM on complex objects.

        Args:
            request: Generation request with prompt and settings

        Returns:
            Tuple of (meshes, images_edited, images_without_background)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        # Qwen uses its own torch.Generator per call, but re-seed global state
        # before each step to ensure determinism regardless of consumed RNG ops.
        set_random_seed(request.seed)
        images_edited = list(self._edit_images(image, request.seed))

        # 2. Remove background
        set_random_seed(request.seed)
        images_with_background = list(image.copy() for image in images_edited)
        images_without_background = self.rmbg.remove_background(images_with_background)

        # 3. Score edit candidates against original and pick the best
        if self.dino_scorer and len(images_edited) > 1 and not self.settings.trellis.multiview:
            logger.info(f"DINO scoring {len(images_edited)} edit candidates against original...")
            self.dino_scorer.dino_model.cuda()
            best_edit_idx, edit_scores = self.dino_scorer.score_images(
                reference_image=image,
                candidate_images=images_without_background,
            )
            logger.success(f"DINO edit champion: candidate {best_edit_idx} (seed={request.seed + best_edit_idx})")

            # Keep only the best candidate
            images_edited = [images_edited[best_edit_idx]]
            images_without_background = [images_without_background[best_edit_idx]]

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        num_candidates = self.settings.trellis.num_candidates
        voxel_thresholds = self.settings.trellis.voxel_complexity_thresholds

        # 3. Phase 1: Generate shapes for all candidates
        # Re-seed so Trellis starts from a known state
        set_random_seed(request.seed)
        trellis_request = TrellisRequest(
            image=images_without_background,
            seed=request.seed,
            num_candidates=num_candidates,
            params=trellis_params
        )
        shape_result = self.trellis.generate_shape(trellis_request)

        # If multi-view fell back to full generation, return directly
        if shape_result.get("is_complete", False):
            return shape_result["meshes"], images_edited, images_without_background

        # Phase 2: Measure voxel complexity and decide texture candidate count
        voxel_count = shape_result.get("voxel_count", 0)
        texture_candidates = num_candidates

        # Walk thresholds in ascending candidate order (1, 2, 3, ...)
        # Pick the lowest candidate count whose threshold is exceeded
        for cand_count in sorted(voxel_thresholds.keys()):
            thresh = voxel_thresholds[cand_count]
            if voxel_count > thresh and cand_count < texture_candidates:
                texture_candidates = cand_count

        if texture_candidates < num_candidates:
            logger.warning(
                f"Voxel complexity {voxel_count} tokens/candidate, "
                f"reducing texture candidates from {num_candidates} to {texture_candidates}"
            )

        # Phase 3: Generate texture with adjusted candidate count
        meshes = self.trellis.generate_texture(shape_result, texture_candidates)

        return meshes, images_edited, images_without_background

    def convert_mesh_to_glb(self, mesh: MeshWithVoxel, glbconv_params: GLBConverterParams) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        glb_mesh = self.glb_converter.convert(mesh, params=glbconv_params)

        buffer = io.BytesIO()
        glb_mesh.export(file_obj=buffer, file_type="glb", extension_webp=False)
        buffer.seek(0)
        
        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return buffer.getvalue()

    def prepare_outputs(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        glb_trellis_result: Optional[TrellisResult]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.

        Args:
            images_edited: List of edited images
            images_without_background: List of images with background removed
            glb_trellis_result: Generated GLB result (optional)

        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        start_time = time.time()
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        # Save generated files if configured
        if self.settings.output.save_generated_files:
            save_files(glb_trellis_result, image_edited_grid, image_without_background_grid)

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.output.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_grid)
            image_without_background_base64 = to_png_base64(image_without_background_grid)
            
        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Execute full generation pipeline with batch output.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets (first candidate GLB + all candidate renders)
        """
        t1 = time.time()
        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate meshes (batch) and get processed images
        meshes, images_edited, images_without_background = await self.generate_meshes(request)

        self._clean_gpu_memory()

        # Convert each mesh to GLB and render as candidate_x.png
        all_glb_bytes = []
        candidate_views = []

        for i in range(len(meshes)):
            mesh = meshes[i]
            meshes[i] = None  # Release reference to save VRAM

            logger.info(f"Processing candidate {i}/{len(meshes) - 1}...")
            mesh.simplify()

            elapsed = time.time() - t1
            dynamic_params = self._get_dynamic_glb_params(mesh, request.glbconv_params, elapsed) if self.settings.api.dynamic_params else request.glbconv_params
            glb_bytes = self.convert_mesh_to_glb(mesh, dynamic_params)
            all_glb_bytes.append(glb_bytes)

            # Render and save as candidate_x.png
            if self.renderer:
                grid_view_bytes = self.renderer.grid_from_glb_bytes(glb_bytes)
                if grid_view_bytes:
                    candidate_path = self.settings.output.output_dir / f"candidate_{i}.png"
                    with open(candidate_path, "wb") as f:
                        f.write(grid_view_bytes)
                    logger.info(f"Rendered candidate_{i}.png saved to {candidate_path}")
                    candidate_views.append(grid_view_bytes)
                else:
                    logger.warning(f"Grid view rendering failed for candidate {i}")

            del mesh, glb_bytes
            self._clean_gpu_memory()

        # VLLM Judge: select best candidate via single-elimination tournament
        best_idx = 0
        if self.duel_manager and len(candidate_views) > 1:
            logger.info(f"VLLM judging {len(candidate_views)} candidates...")
            prompt_image_bytes = base64.b64decode(request.prompt_image)

            for i in range(1, len(candidate_views)):
                logger.info(f"Judging candidate {i} vs champion (candidate {best_idx})...")
                winner_idx, issues = await self.duel_manager.run_duel(
                    prompt_image_bytes,
                    candidate_views[best_idx],
                    candidate_views[i],
                    request.seed,
                )
                if winner_idx == 1:  # Challenger won
                    logger.info(f"Candidate {i} wins over candidate {best_idx}")
                    best_idx = i
                else:
                    logger.info(f"Candidate {best_idx} retains champion status")

            logger.success(f"Champion: candidate {best_idx}")

        glb_trellis_result = TrellisResult(file_bytes=all_glb_bytes[best_idx]) if all_glb_bytes else None
        del all_glb_bytes

        # Save generated files
        image_edited_base64, image_no_bg_base64 = None, None
        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited_base64, image_no_bg_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                glb_trellis_result
            )
        del images_edited, images_without_background

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s ({len(candidate_views)} candidates rendered, best={best_idx})")

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            grid_view_file_base64=candidate_views[best_idx] if candidate_views else None,
            candidate_views=candidate_views if candidate_views else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64
        )

        return response