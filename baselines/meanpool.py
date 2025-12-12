"""Embedding-level mean pooling compression (Lee et al. replication).

This module implements mean pooling compression at the embedding level,
replicating the approach from:
    Lee et al., "Optical Context Compression Is Just (Bad) Autoencoding"
    https://github.com/ivnle/bad-autoencoding/blob/main/trainers/meanpool.py

The key insight is that this operates on neural representations directly,
not on text tokens, providing a fair comparison to vision-based compression.

Trained checkpoints available from HuggingFace:
    https://huggingface.co/ivnle/bad-autoencoding
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from .config import GUNDAM_PRESET
from .utils.model import get_device, load_model

logger = logging.getLogger(__name__)

# Lee et al.'s trained checkpoint configuration
LEE_CHECKPOINT_REPO = "ivnle/bad-autoencoding"
LEE_CHECKPOINT_FILE = "meanpool_w4s4_h0_lm_recon-init/best_checkpoint.pt"
LEE_WINDOW_SIZE = 4
LEE_STRIDE = 4


def download_lee_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Download Lee et al.'s trained mean pooling checkpoint.

    Args:
        cache_dir: Optional cache directory for the checkpoint.

    Returns:
        Path to the downloaded checkpoint file.
    """
    logger.info(f"Downloading trained checkpoint from {LEE_CHECKPOINT_REPO}...")
    checkpoint_path = hf_hub_download(
        repo_id=LEE_CHECKPOINT_REPO,
        filename=LEE_CHECKPOINT_FILE,
        cache_dir=cache_dir,
    )
    logger.info(f"Checkpoint downloaded to: {checkpoint_path}")
    return Path(checkpoint_path)


def load_trained_meanpool_model(
    cache_dir: str | Path | None = None,
) -> tuple[AutoModel, AutoTokenizer, torch.Tensor]:
    """Load DeepSeek-OCR with Lee et al.'s trained mean pooling weights.

    This loads:
    1. The base DeepSeek-OCR model
    2. Lee et al.'s trained checkpoint (fine-tuned decoder + separator)

    Args:
        cache_dir: Optional cache directory for the checkpoint.

    Returns:
        Tuple of (model, tokenizer, separator_embed)
    """
    # Load base model
    model, tokenizer = load_model()
    device = get_device()

    # Download and load checkpoint
    checkpoint_path = download_lee_checkpoint(cache_dir)
    logger.info("Loading checkpoint weights...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Extract the trained separator embedding
    separator_key = "meanpool_separator"
    if separator_key in state_dict:
        separator_embed = state_dict.pop(separator_key)
        logger.info(f"Loaded trained separator embedding: {separator_embed.shape}")
    else:
        logger.warning("No trained separator found in checkpoint, using random init")
        hidden_dim = model.config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        separator_embed = torch.randn(hidden_dim, dtype=torch.bfloat16) * embed_std

    # Load remaining weights into model
    # Filter to only load compatible keys
    model_state = model.state_dict()
    compatible_keys = {k: v for k, v in state_dict.items() if k in model_state}
    incompatible = set(state_dict.keys()) - set(compatible_keys.keys()) - {separator_key}

    if incompatible:
        logger.warning(f"Skipping {len(incompatible)} incompatible keys from checkpoint")

    model.load_state_dict(compatible_keys, strict=False)
    logger.info(f"Loaded {len(compatible_keys)} weight tensors from checkpoint")

    return model, tokenizer, separator_embed.to(device)


class EmbeddingMeanPooler:
    """
    Embedding-level mean pooling compression.

    This compresses text by:
    1. Getting token embeddings from the model's embedding layer
    2. Applying sliding window mean pooling in embedding space
    3. Injecting pooled embeddings back into the model via masked_scatter_()

    Unlike text-level approximations, this operates on actual neural representations.

    When use_trained=True, uses Lee et al.'s trained checkpoint with:
    - Fine-tuned decoder weights
    - Trained separator embedding
    - Fixed window=4, stride=4 configuration
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        target_tokens: int = 400,
        device: str = "cuda",
        separator_embed: torch.Tensor | None = None,
        window_size: int | None = None,
        stride: int | None = None,
    ):
        """
        Initialize embedding mean pooler.

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            target_tokens: Target number of compressed tokens (including separator)
            device: Device to use
            separator_embed: Pre-trained separator embedding (optional)
            window_size: Fixed window size (optional, overrides dynamic calculation)
            stride: Fixed stride (optional, overrides dynamic calculation)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.device = torch.device(device)

        # Fixed window/stride for trained models
        self.fixed_window_size = window_size
        self.fixed_stride = stride

        # Get model's hidden dimension
        self.hidden_dim = model.config.hidden_size

        # IMAGE_TOKEN_ID from DeepSeek-OCR (placeholder for injected embeddings)
        self.placeholder_token_id = tokenizer.encode("<image>", add_special_tokens=False)[0]

        # BOS token
        self.bos_token_id = tokenizer.bos_token_id

        # Use provided separator or create random one
        if separator_embed is not None:
            self.separator_embed = separator_embed.to(self.device)
            self.is_trained = True
            logger.info("[EmbeddingMeanPooler] Using trained separator embedding")
        else:
            # Create separator embedding (following Lee et al.)
            # For inference-only, we use random initialization
            embed_std = 1 / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
            self.separator_embed = torch.randn(
                self.hidden_dim, device=self.device, dtype=torch.bfloat16
            ) * embed_std
            self.is_trained = False
            logger.info("[EmbeddingMeanPooler] Using random separator (untrained)")

        # Pre-allocate zero images for vision bypass
        self._setup_vision_bypass()

        # Log configuration
        logger.info(f"[EmbeddingMeanPooler] Target tokens: {target_tokens}")
        logger.info(f"[EmbeddingMeanPooler] Hidden dim: {self.hidden_dim}")
        if self.fixed_window_size:
            logger.info(f"[EmbeddingMeanPooler] Fixed window: {window_size}, stride: {stride}")

    def _setup_vision_bypass(self):
        """Setup dummy images to bypass vision encoder."""
        base_size = GUNDAM_PRESET["base_size"]
        image_size = GUNDAM_PRESET["image_size"]
        self.empty_crop = torch.zeros(
            0, 3, image_size, image_size,
            dtype=torch.bfloat16, device=self.device
        )
        self.zero_global = torch.zeros(
            1, 3, base_size, base_size,
            dtype=torch.bfloat16, device=self.device
        )

    def _calculate_window_params(self, context_length: int) -> tuple[int, int]:
        """Calculate window size and stride for target compression.

        Args:
            context_length: Number of context tokens

        Returns:
            Tuple of (window_size, stride)
        """
        # Use fixed values if provided (e.g., for trained models)
        if self.fixed_window_size is not None and self.fixed_stride is not None:
            return self.fixed_window_size, self.fixed_stride

        # Target: compress context_length tokens to target_tokens
        # We need (target_tokens - 1) pooled windows + 1 separator
        num_windows = self.target_tokens - 1

        if num_windows <= 0:
            raise ValueError(f"target_tokens must be > 1, got {self.target_tokens}")

        # Calculate window size and stride for even coverage
        # Using non-overlapping windows (stride = window_size)
        window_size = max(1, context_length // num_windows)
        stride = window_size

        return window_size, stride

    def _sliding_window_mean_pool(
        self, embeds: torch.Tensor, window_size: int, stride: int
    ) -> torch.Tensor:
        """
        Apply sliding window mean pooling to embeddings.

        Args:
            embeds: Token embeddings [batch_size, seq_len, hidden_dim]
            window_size: Size of sliding window
            stride: Stride between windows

        Returns:
            Pooled embeddings [batch_size, num_windows, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = embeds.shape

        if seq_len < window_size:
            # Context too short, just mean pool everything
            return embeds.mean(dim=1, keepdim=True)

        # Use unfold to extract windows
        windows = embeds.unfold(1, window_size, stride)
        # Shape: [batch_size, num_windows, hidden_dim, window_size]

        # Mean pool each window
        pooled_regular = windows.mean(dim=-1)
        # Shape: [batch_size, num_windows, hidden_dim]

        # Handle remainder tokens (flexible last window)
        num_regular = pooled_regular.shape[1]
        regular_end_pos = (num_regular - 1) * stride + window_size

        if regular_end_pos < seq_len:
            # Pool remaining tokens
            remainder = embeds[:, regular_end_pos:, :]
            pooled_remainder = remainder.mean(dim=1, keepdim=True)
            pooled = torch.cat([pooled_regular, pooled_remainder], dim=1)
        else:
            pooled = pooled_regular

        return pooled

    def compress_and_generate(
        self,
        context_text: str,
        prompt_text: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Compress context via mean pooling and generate response.

        Args:
            context_text: The context to compress
            prompt_text: The prompt/question to answer
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Tokenize context
        context_tokens = self.tokenizer.encode(
            context_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        context_length = context_tokens.shape[1]

        # Calculate window parameters
        window_size, stride = self._calculate_window_params(context_length)

        # Get context embeddings
        with torch.no_grad():
            context_embeds = self.model.model.get_input_embeddings()(context_tokens)

            # Apply mean pooling
            pooled_embeds = self._sliding_window_mean_pool(
                context_embeds, window_size, stride
            )
            num_pooled = pooled_embeds.shape[1]

            # Add separator
            separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
            pooled_with_sep = torch.cat([pooled_embeds, separator], dim=1)
            num_compressed = num_pooled + 1

            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)

            # Build input sequence: [BOS] + [POOLED_PLACEHOLDERS] + [PROMPT]
            batch_size = 1
            bos = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
            placeholders = torch.full(
                (batch_size, num_compressed), self.placeholder_token_id, dtype=torch.long, device=self.device
            )
            input_ids = torch.cat([bos, placeholders, prompt_tokens], dim=1)

            # Get initial embeddings
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

            # Create mask for pooled positions
            mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=self.device)
            mask[:, 1:1+num_compressed] = True

            # Inject pooled embeddings via masked_scatter_
            inputs_embeds.masked_scatter_(
                mask.unsqueeze(-1),
                pooled_with_sep.reshape(-1, self.hidden_dim)
            )

            # Prepare vision bypass
            images = [(self.empty_crop, self.zero_global)]
            images_spatial_crop = [[1, 1]]

            # Generate with manual autoregressive decoding
            generated_tokens = []
            current_input_ids = input_ids
            current_embeds = inputs_embeds

            for _ in range(max_new_tokens):
                current_seq_len = current_embeds.shape[1]
                current_images_seq_mask = torch.zeros(
                    batch_size, current_seq_len, dtype=torch.bool, device=self.device
                )

                outputs = self.model.forward(
                    input_ids=current_input_ids,
                    inputs_embeds=current_embeds,
                    images=images,
                    images_spatial_crop=images_spatial_crop,
                    images_seq_mask=current_images_seq_mask,
                    use_cache=False,
                    return_dict=True,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token.item())

                next_token_2d = next_token.unsqueeze(1)
                next_embed = self.model.model.get_input_embeddings()(next_token_2d)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                current_input_ids = torch.cat([current_input_ids, next_token_2d], dim=1)

            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return output_text


def run_inference_mean_pool(
    prompt: str,
    context: str,
    target_tokens: int = 400,
    model: AutoModel | None = None,
    tokenizer: AutoTokenizer | None = None,
    use_trained: bool = False,
    separator_embed: torch.Tensor | None = None,
) -> tuple[str, int, int]:
    """
    Run inference using embedding-level mean pooling compression.

    Args:
        prompt: The question/prompt to answer
        context: The context text to compress
        target_tokens: Target number of compressed tokens
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        use_trained: If True, load Lee et al.'s trained checkpoint
        separator_embed: Pre-loaded trained separator (optional)

    Returns:
        Tuple of (output_text, compressed_tokens, output_tokens)
    """
    if use_trained and model is None:
        # Load trained model with separator
        model, tokenizer, separator_embed = load_trained_meanpool_model()
        window_size = LEE_WINDOW_SIZE
        stride = LEE_STRIDE
    elif model is None or tokenizer is None:
        model, tokenizer = load_model()
        window_size = None
        stride = None
    else:
        window_size = LEE_WINDOW_SIZE if separator_embed is not None else None
        stride = LEE_STRIDE if separator_embed is not None else None

    pooler = EmbeddingMeanPooler(
        model=model,
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        device=get_device(),
        separator_embed=separator_embed,
        window_size=window_size,
        stride=stride,
    )

    output = pooler.compress_and_generate(
        context_text=context,
        prompt_text=prompt,
        max_new_tokens=50,
    )

    output_tokens = len(tokenizer.encode(output, add_special_tokens=False))

    return output, target_tokens, output_tokens
