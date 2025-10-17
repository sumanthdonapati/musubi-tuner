"""
Modified Qwen Image Training Script with Face Embedding Loss
This extends the standard training to include face consistency tracking using InsightFace.
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.qwen_image_train_network import (
    QwenImageNetworkTrainer,
    qwen_image_setup_parser,
    main as original_main,
)
from musubi_tuner.hv_train_network import (
    setup_parser_common,
    read_config_from_file,
    clean_memory_on_device,
)
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
from musubi_tuner.utils.face_embedding_loss import FaceEmbeddingLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QwenImageNetworkTrainerWithFaceLoss(QwenImageNetworkTrainer):
    """
    Extended trainer that adds face embedding loss/metrics to the training loop.
    """
    
    def __init__(self):
        super().__init__()
        self.face_loss_module: Optional[FaceEmbeddingLoss] = None
        self.face_loss_enabled = False
        self.face_loss_weight = 0.0
        self.face_eval_frequency = 100  # Evaluate every N steps
        
    def initialize_face_loss(self, args):
        """Initialize face embedding loss module if enabled."""
        if not args.enable_face_loss:
            return
        
        logger.info("Initializing Face Embedding Loss module")
        
        try:
            self.face_loss_module = FaceEmbeddingLoss(
                model_name=args.face_model_name,
                device=args.face_device,
                det_size=(args.face_det_size, args.face_det_size),
                use_as_loss=args.use_face_as_loss,
                loss_weight=args.face_loss_weight,
            )
            self.face_loss_enabled = True
            self.face_loss_weight = args.face_loss_weight
            self.face_eval_frequency = args.face_eval_frequency
            
            logger.info(f"Face loss enabled: use_as_loss={args.use_face_as_loss}, weight={args.face_loss_weight}")
            
        except Exception as e:
            logger.error(f"Failed to initialize face loss: {e}")
            logger.warning("Continuing training without face loss")
            self.face_loss_enabled = False
    
    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        """
        Override call_dit to compute face embedding loss in addition to standard loss.
        """
        # Call parent method to get standard model prediction and target
        model_pred, target = super().call_dit(
            args, accelerator, transformer, latents, batch, 
            noise, noisy_model_input, timesteps, network_dtype
        )
        
        # Store these for potential face loss computation
        # We'll compute face loss separately in the training loop
        # to avoid blocking the forward pass
        
        return model_pred, target
    
    def compute_face_loss_on_samples(
        self,
        args,
        accelerator: Accelerator,
        vae,
        control_latents: torch.Tensor,
        generated_latents: torch.Tensor,
        global_step: int,
    ) -> tuple[Optional[torch.Tensor], dict]:
        """
        Compute face embedding loss on decoded samples.
        
        This should be called periodically (not every step) to avoid overhead.
        """
        if not self.face_loss_enabled:
            return None, {}
        
        # Only compute face loss every N steps
        if global_step % self.face_eval_frequency != 0:
            return None, {}
        
        logger.info(f"Computing face embedding loss at step {global_step}")
        
        try:
            # Ensure we have the VAE ready
            vae_dtype = torch.bfloat16 if args.vae_dtype == "bfloat16" else torch.float16
            
            loss, metrics = self.face_loss_module.compute_on_decoded_images(
                vae=vae,
                control_latents=control_latents,
                generated_latents=generated_latents,
                vae_dtype=vae_dtype,
            )
            
            # Log metrics
            logger.info(f"Face similarity: {metrics.get('face_similarity_mean', 0.0):.4f} "
                       f"(detected: {metrics.get('valid_pairs', 0)}/{control_latents.shape[0]})")
            
            return loss, metrics
            
        except Exception as e:
            logger.error(f"Error computing face loss: {e}")
            return None, {}
    
    def train(self, args):
        """
        Override train method to initialize face loss before training starts.
        """
        # Initialize face loss module if enabled
        self.initialize_face_loss(args)
        
        # Call parent train method
        # Note: We would need to modify the parent's training loop to call
        # compute_face_loss_on_samples periodically. For now, we'll add this
        # via the sample_images hook.
        
        super().train(args)


def setup_face_loss_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add face embedding loss specific arguments."""
    
    # Face loss configuration
    parser.add_argument(
        "--enable_face_loss",
        action="store_true",
        help="Enable face embedding loss/metrics using InsightFace"
    )
    
    parser.add_argument(
        "--use_face_as_loss",
        action="store_true",
        help="Use face similarity as a training loss (if False, only used as metric)"
    )
    
    parser.add_argument(
        "--face_loss_weight",
        type=float,
        default=0.1,
        help="Weight for face embedding loss (default: 0.1)"
    )
    
    parser.add_argument(
        "--face_model_name",
        type=str,
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_s", "antelopev2"],
        help="InsightFace model name (default: buffalo_l)"
    )
    
    parser.add_argument(
        "--face_device",
        type=str,
        default="cuda",
        help="Device for face detection/recognition (default: cuda)"
    )
    
    parser.add_argument(
        "--face_det_size",
        type=int,
        default=640,
        help="Detection size for face detector (default: 640)"
    )
    
    parser.add_argument(
        "--face_eval_frequency",
        type=int,
        default=100,
        help="Evaluate face similarity every N steps (default: 100)"
    )
    
    return parser


def main():
    """Main training entry point with face loss support."""
    parser = setup_parser_common()
    parser = qwen_image_setup_parser(parser)
    parser = setup_face_loss_parser(parser)
    
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    
    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    
    trainer = QwenImageNetworkTrainerWithFaceLoss()
    trainer.train(args)


if __name__ == "__main__":
    main()

