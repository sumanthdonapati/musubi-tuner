"""
Face Embedding Loss Module using InsightFace
This module computes face similarity between control and generated images
for tracking face consistency during training.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from PIL import Image

logger = logging.getLogger(__name__)


class FaceEmbeddingLoss:
    """
    Computes face embedding similarity loss using InsightFace models.
    
    This can be used as:
    1. A training loss component (added to the main loss)
    2. A validation metric (logged during training)
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
        use_as_loss: bool = False,
        loss_weight: float = 0.1,
    ):
        """
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, antelopev2)
            device: Device to run face detection/recognition
            det_size: Detection size for face detector
            use_as_loss: If True, return loss value; if False, return similarity metric
            loss_weight: Weight for face embedding loss when use_as_loss=True
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace not installed. Install with: pip install insightface onnxruntime"
            )
        
        self.device = device
        self.use_as_loss = use_as_loss
        self.loss_weight = loss_weight
        
        # Initialize InsightFace
        logger.info(f"Initializing InsightFace with model: {model_name}")
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)
        logger.info("InsightFace initialized successfully")
        
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3) with values in [0, 255]
            
        Returns:
            Face embedding vector (512-dim) or None if no face detected
        """
        try:
            faces = self.app.get(image)
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
            
            # Use the face with largest bounding box (most prominent)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            embedding = largest_face.embedding  # (512,) normalized vector
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity in range [-1, 1], higher is more similar
        """
        # Embeddings from InsightFace are already L2 normalized
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array in correct format for InsightFace.
        
        Args:
            tensor: Image tensor (C, H, W) or (B, C, H, W) with values in [-1, 1] or [0, 1]
            
        Returns:
            Numpy array (H, W, 3) with values in [0, 255] uint8 (RGB)
        """
        # Handle batch dimension
        if tensor.ndim == 4:
            tensor = tensor[0]  # Take first image in batch
        
        # Move to CPU and convert to numpy
        img = tensor.detach().cpu().float()
        
        # Denormalize if needed (assume [-1, 1] or [0, 1] range)
        if img.min() < 0:
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Clamp to valid range
        img = torch.clamp(img, 0, 1)
        
        # Convert to HWC format and scale to [0, 255]
        img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        img = (img * 255).astype(np.uint8)
        
        # Ensure RGB format (InsightFace expects RGB)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        
        return img
    
    def compute_face_consistency_loss(
        self,
        control_images: torch.Tensor,
        generated_images: torch.Tensor,
        return_metrics: bool = False,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Compute face consistency loss between control and generated images.
        
        Args:
            control_images: Control/reference images tensor (B, C, H, W)
            generated_images: Generated images tensor (B, C, H, W)
            return_metrics: If True, return detailed metrics
            
        Returns:
            loss: Loss tensor (if use_as_loss=True) or None
            metrics: Dictionary with similarity scores and detection info
        """
        batch_size = control_images.shape[0]
        similarities = []
        detected_pairs = 0
        
        metrics = {
            "face_similarity_mean": 0.0,
            "face_similarity_min": 1.0,
            "face_similarity_max": 0.0,
            "faces_detected_control": 0,
            "faces_detected_generated": 0,
            "valid_pairs": 0,
        }
        
        for i in range(batch_size):
            # Convert tensors to numpy
            control_img = self.tensor_to_numpy(control_images[i])
            generated_img = self.tensor_to_numpy(generated_images[i])
            
            # Extract embeddings
            control_emb = self.extract_face_embedding(control_img)
            generated_emb = self.extract_face_embedding(generated_img)
            
            # Update detection counts
            if control_emb is not None:
                metrics["faces_detected_control"] += 1
            if generated_emb is not None:
                metrics["faces_detected_generated"] += 1
            
            # Compute similarity if both faces detected
            if control_emb is not None and generated_emb is not None:
                similarity = self.compute_similarity(control_emb, generated_emb)
                similarities.append(similarity)
                detected_pairs += 1
        
        # Compute statistics
        if len(similarities) > 0:
            metrics["face_similarity_mean"] = float(np.mean(similarities))
            metrics["face_similarity_min"] = float(np.min(similarities))
            metrics["face_similarity_max"] = float(np.max(similarities))
            metrics["valid_pairs"] = detected_pairs
            
            # Convert to loss if needed
            if self.use_as_loss:
                # Loss = 1 - similarity (minimize distance, maximize similarity)
                similarity_tensor = torch.tensor(
                    similarities, 
                    dtype=torch.float32,
                    device=control_images.device
                )
                loss = (1.0 - similarity_tensor).mean() * self.loss_weight
                return loss, metrics
        else:
            logger.warning(f"No valid face pairs detected in batch of {batch_size}")
        
        return None, metrics
    
    def compute_on_decoded_images(
        self,
        vae,
        control_latents: torch.Tensor,
        generated_latents: torch.Tensor,
        vae_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Decode latents and compute face consistency loss.
        
        Args:
            vae: VAE model for decoding
            control_latents: Control image latents (B, C, 1, H, W)
            generated_latents: Generated image latents (B, C, 1, H, W)
            vae_dtype: Data type for VAE computation
            
        Returns:
            loss: Loss tensor or None
            metrics: Dictionary with metrics
        """
        original_device = vae.device
        vae.to(self.device)
        vae.eval()
        
        try:
            with torch.no_grad():
                # Decode control images
                control_images = vae.decode(control_latents.to(self.device, vae_dtype))
                control_images = control_images.sample if hasattr(control_images, 'sample') else control_images
                
                # Decode generated images
                generated_images = vae.decode(generated_latents.to(self.device, vae_dtype))
                generated_images = generated_images.sample if hasattr(generated_images, 'sample') else generated_images
            
            # Compute face consistency
            loss, metrics = self.compute_face_consistency_loss(
                control_images, 
                generated_images,
                return_metrics=True
            )
            
            return loss, metrics
            
        finally:
            vae.to(original_device)
    
    @torch.no_grad()
    def evaluate_on_image_files(
        self,
        control_image_paths: List[str],
        generated_image_paths: List[str],
    ) -> dict:
        """
        Evaluate face consistency on saved image files.
        
        Args:
            control_image_paths: List of control image file paths
            generated_image_paths: List of generated image file paths
            
        Returns:
            Dictionary with evaluation metrics
        """
        assert len(control_image_paths) == len(generated_image_paths)
        
        similarities = []
        
        for ctrl_path, gen_path in zip(control_image_paths, generated_image_paths):
            # Load images
            ctrl_img = np.array(Image.open(ctrl_path).convert('RGB'))
            gen_img = np.array(Image.open(gen_path).convert('RGB'))
            
            # Extract embeddings
            ctrl_emb = self.extract_face_embedding(ctrl_img)
            gen_emb = self.extract_face_embedding(gen_img)
            
            if ctrl_emb is not None and gen_emb is not None:
                sim = self.compute_similarity(ctrl_emb, gen_emb)
                similarities.append(sim)
        
        if len(similarities) > 0:
            return {
                "face_similarity_mean": float(np.mean(similarities)),
                "face_similarity_std": float(np.std(similarities)),
                "face_similarity_min": float(np.min(similarities)),
                "face_similarity_max": float(np.max(similarities)),
                "valid_samples": len(similarities),
                "total_samples": len(control_image_paths),
            }
        else:
            return {
                "face_similarity_mean": 0.0,
                "valid_samples": 0,
                "total_samples": len(control_image_paths),
            }

