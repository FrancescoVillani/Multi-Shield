import torch


class MultiShield(torch.nn.Module):
    """
    Multi-Shield defense combining DNN classifier with CLIP model.
    
    Multi-Shield operates in three phases:
    1. Unimodal classification: DNN classifier makes initial prediction
    2. Multi-modal alignment: CLIP model computes alignment scores
    3. Decision: If both models agree, output prediction; otherwise, reject
    
    The rejection is implemented as an additional output dimension where a high
    score indicates that the sample should be rejected (models disagree).
    
    Args:
        dnn: Trained DNN classifier (e.g., from RobustBench)
        clip_model: ClipModel instance configured for the dataset
        tolerance: Tolerance threshold for agreement (default: 0.001)
    
    Example:
        >>> model = get_local_model("carmon2019", "cifar10", normalize)
        >>> clip = ClipModel(...)
        >>> multi_shield = MultiShield(dnn=model, clip_model=clip)
        >>> outputs = multi_shield(images)  # Shape: [batch, num_classes + 1]
    """
    
    def __init__(self, dnn, clip_model, tolerance=0.001):
        super(MultiShield, self).__init__()
        self.dnn = dnn
        self.clip = clip_model
        self.tolerance = tolerance

    def forward(self, inputs):
        """
        Forward pass through Multi-Shield.
        
        Args:
            inputs: Batch of input images [batch, channels, height, width]
        
        Returns:
            Tensor of shape [batch, num_classes + 1] where the last dimension
            is the rejection score. Higher rejection scores indicate the sample
            should be rejected (DNN and CLIP disagree).
        """
        # Phase 1: Get DNN predictions
        dnn_raw_predictions = self.dnn(inputs)
        dnn_predicted_labels = dnn_raw_predictions.argmax(dim=-1)

        # Phase 2: Get CLIP embeddings and compute alignment
        image_encoding = self.clip.create_image_embedding(inputs)
        cosine_similarity = self.clip.cosine_similarity(
            image_encoding, self.clip.label_emb
        )

        # Find max CLIP score and score for DNN's predicted class
        cos_sim_max, _ = torch.max(cosine_similarity, dim=1)
        cosine_i = cosine_similarity[
            torch.arange(cosine_similarity.size(0)), dnn_predicted_labels
        ]

        # Phase 3: Compute rejection score
        # High rejection score = DNN and CLIP disagree
        rejection_score = (
            torch.max(dnn_raw_predictions, dim=1)[0]
            + torch.abs(cos_sim_max - cosine_i)
            - self.tolerance
        )

        # Return predictions with rejection score as additional dimension
        return torch.cat((dnn_raw_predictions, rejection_score.unsqueeze(1)), dim=1)
