"""
CLIP Model Wrapper for Multi-Shield

This module provides a wrapper around CLIP models (from HuggingFace Transformers
or OpenCLIP) for zero-shot image classification.
"""

from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
import torch
import torch.nn as nn
import open_clip
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()

# Prompt templates for different datasets
PROMPT_TEMPLATES = {
    "cifar10": "this is a photo of a {}",
    "mnist": "Digit {}",
    "default": "photo of a {}"
}


class ClipModel(nn.Module):
    """
    CLIP model wrapper for zero-shot classification.
    
    Supports both HuggingFace Transformers and OpenCLIP implementations.
    """

    def __init__(
        self,
        model_name: str,
        processor_name: str,
        tokenizer_name: str,
        use_open_clip: bool,
        label_names: list[str],
        torch_preprocess,
        dataset: str,
        device: torch.device,
        resize=None
    ):
        super().__init__()
        
        self.instantiate_model(
            model_name, processor_name, tokenizer_name, use_open_clip, torch_preprocess
        )
        self.device = device
        self.model.to(self.device)
        self.dataset = dataset
        self.labels = label_names
        
        # Use prompt templates
        template = PROMPT_TEMPLATES.get(dataset, PROMPT_TEMPLATES["default"])
        self.clip_labels = [template.format(label) for label in self.labels]
        
        self.instantiate_label_embeddings()
        self.resize = resize

    def instantiate_model(
        self,
        model_name: str,
        processor_name: str,
        tokenizer_name: str,
        use_open_clip: bool,
        torch_preprocess=None,
    ) -> None:
        """Initialize the CLIP model from either OpenCLIP or HuggingFace."""
        if use_open_clip:
            model, _, processor = open_clip.create_model_and_transforms(model_name)
            model.eval()
            tokenizer = open_clip.get_tokenizer(tokenizer_name)
        else:
            processor = CLIPProcessor.from_pretrained(processor_name)
            vision_model = CLIPVisionModel.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(processor_name)
            model.vision_model.load_state_dict(vision_model.vision_model.state_dict())
            tokenizer = None

        self.use_open_clip = use_open_clip
        self.torch_processor = torch_preprocess
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        model.eval()

    def instantiate_label_embeddings(self) -> None:
        """Compute and cache text embeddings for all class labels."""
        if self.use_open_clip:
            text = self.tokenizer(self.clip_labels)
            text_features = self.model.encode_text(text.to(self.device))
            text_features = F.normalize(text_features, dim=-1)
            self.label_emb = text_features
        else:
            self.label_tokens = self.processor(
                text=self.clip_labels, padding=True, images=None, return_tensors="pt"
            ).to(self.device)

            self.label_emb = self.model.get_text_features(**self.label_tokens)
            self.label_emb = self.label_emb.to(self.device)
            # Normalize embeddings properly (dim=1 for feature dimension)
            self.label_emb = F.normalize(self.label_emb, p=2, dim=1)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            a: First set of vectors
            b: Second set of vectors
        
        Returns:
            Cosine similarity matrix
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def create_image_embedding(self, batch_image: torch.Tensor) -> torch.Tensor:
        """
        Create CLIP image embeddings for a batch of images.
        
        Args:
            batch_image: Batch of images (batch_size, channels, height, width)
        
        Returns:
            Image embeddings (batch_size, embedding_dim)
        """
        if self.resize is not None:
            batch_image = self.resize(batch_image)

        if self.use_open_clip:
            if self.torch_processor is not None:
                processed_images = torch.stack(
                    [self.torch_processor(image) for image in batch_image]
                )
            else:
                to_pil = ToPILImage()
                processed_images = torch.stack(
                    [self.processor(to_pil(image)) for image in batch_image]
                )

            image = processed_images.to(self.device)
            image_features = self.model.encode_image(image)
            image_emb = F.normalize(image_features, dim=-1)
        else:
            processed_images = torch.stack(
                [self.torch_processor(image) for image in batch_image]
            )
            if self.resize is not None:
                processed_images = self.resize(processed_images)
            image_emb = self.model.get_image_features(processed_images)
            image_emb = image_emb.to(self.device)
        return image_emb

    def clip_prediction(self, image_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using CLIP embeddings.
        
        Args:
            image_emb: Image embeddings
            labels: Ground truth labels
        
        Returns:
            Binary tensor indicating correct predictions
        """
        labels = labels.to(self.device)

        scores = torch.mm(image_emb, self.label_emb.transpose(0, 1))
        predictions = torch.argmax(scores, dim=1)
        correct_predictions = (predictions == labels).float()

        return correct_predictions


class ClipClassifier(nn.Module):
    """
    Standalone CLIP classifier that can be used as a drop-in replacement
    for standard classifiers.
    """
    
    def __init__(self, clip_model: ClipModel):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CLIP classifier.
        
        Args:
            x: Input images
        
        Returns:
            Cosine similarity scores for each class
        """
        image_encoding = self.clip.create_image_embedding(x)
        return torch.abs(
            self.clip.cosine_similarity(image_encoding, self.clip.label_emb)
        )
