import torch
from tqdm import tqdm
import numpy as np

from classifiers.single_input.EmbeddingFF import EmbeddingFF
from trainers.BaseFFTrainer import BaseFFTrainer
from torch.nn.functional import cosine_similarity

class EmbeddingFFTrainer(BaseFFTrainer):
    """
    Trainer for models that return processed embeddings from the feature processor and use 
    embedding-based loss functions like AAM.
    
    This trainer expects the model's forward method to return (logits, probs, processed_embeddings),
    and applies the loss function to the processed embeddings from the feature processor.
    """
    
    def __init__(
        self, model: EmbeddingFF, lossfn, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trainer for embedding-based models and losses.
        
        Args:
            model: The model to train (must be EmbeddingFF or similar that returns processed embeddings)
            lossfn: The loss function to use (must operate on embeddings)
            device: The device to use for training
        """
        super().__init__(model, lossfn, device)
        
        # Validate that the model returns embeddings
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 16000).to(device)  # Small dummy input
            outputs = self.model(dummy_input)
            if len(outputs) != 3:
                raise ValueError(
                    "Model must return a tuple of (logits, probs, processed_embeddings) "
                    f"but got {len(outputs)} outputs instead"
                )
        self.model.train()

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model on the given dataloader for one epoch
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        return: Tuple(lists of accuracies, list of losses)
        """
        # For accuracy computation in the epoch
        losses = []
        accuracies = []

        # Training loop
        for _, wf, label in tqdm(train_dataloader):
            wf = wf.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, probs, processed_embeddings = self.model(wf)

            # Use processed embeddings with the loss function
            loss = self.lossfn(processed_embeddings, label.long())
            loss.backward()
            self.optimizer.step()

            # Compute accuracy from the model's classifier output
            predicted = torch.argmax(probs, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss.item())
            accuracies.append(accuracy)

        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[tuple[str, str]]]:
        """
        Validate the model on the given dataloader with paired inputs

        Args:
            val_dataloader: Dataloader loading the validation data
            save_scores: Whether to save the scores
            
        Returns:
            Tuple(losses, predictions, labels, scores, file_names)
            - losses: List of loss values
            - labels: List of ground truth labels (1 for same speaker, 0 for different)
            - scores: List of cosine distances
            - predictions: List of embeddings
            - file_names: List of (source_path, target_path) tuples
        """
        losses = []
        labels = []
        scores = []  # Will store cosine distances
        predictions = []
        file_names = []

        self.model.eval()
        with torch.no_grad():
            for (source_paths, target_paths), (source_wf, target_wf), label in tqdm(val_dataloader):
                # Move data to device
                source_wf = source_wf.to(self.device)
                target_wf = target_wf.to(self.device)
                label = label.to(self.device)

                # Get embeddings for both source and target
                _, _, source_embeddings = self.model(source_wf)
                _, _, target_embeddings = self.model(target_wf)
                
                # Compute cosine similarity
                similarities = cosine_similarity(source_embeddings, target_embeddings)

                predictions.extend(source_embeddings.cpu().tolist())

                if save_scores:
                    file_names.extend(list(zip(source_paths, target_paths)))
            
                labels.extend(label.cpu().tolist())
                scores.extend(similarities.cpu().tolist())

        return losses, labels, scores, predictions, file_names