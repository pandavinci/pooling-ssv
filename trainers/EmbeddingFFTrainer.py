import torch
from tqdm import tqdm
import numpy as np

from classifiers.single_input.EmbeddingFF import EmbeddingFF
from trainers.BaseFFTrainer import BaseFFTrainer
from torch.nn.functional import cosine_similarity

from trainers.utils import trace_handler
from torch.profiler import record_function

class EmbeddingFFTrainer(BaseFFTrainer):
    """
    Trainer for models that return processed embeddings from the feature processor and use
    a loss function that operates on those embeddings rather than classifier outputs.

    This trainer expects the model's forward method to return (logits, probs, processed_embeddings),
    and applies the loss function to the processed embeddings from the feature processor.
    """

    def __init__(
        self, model: EmbeddingFF, device="cuda" if torch.cuda.is_available() else "cpu", save_embeddings=False, save_path=None
    ):
        """
        Initialize the EmbeddingFFTrainer.

        Args:
            model: The model to train (must be EmbeddingFF or similar that returns processed embeddings)
            device: Device to use for training
            save_embeddings: Whether to save embeddings during validation
            save_path: Path prefix for saving all training outputs
        """
        super().__init__(model, device, save_embeddings, save_path)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model on the given dataloader for one epoch
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        return: Tuple(lists of accuracies, list of losses)
        """
        # For accuracy computation in the epoch
        losses = []

        # Training loop
        """ iterations_total = 200
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=iterations_total, repeat=0),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
            on_trace_ready=trace_handler,
        ) as prof:
            iterations = 0 """
        for _, wf, label in tqdm(train_dataloader):
            """ iterations += 1
            if iterations > iterations_total:
                break """
            wf = wf.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            #with record_function("### forward ###"):
            processed_embeddings = self.model(wf)

            # Use processed embeddings with the loss function
            #with record_function("### loss ###"):
            loss = self.model.loss_fn(processed_embeddings, label.long())
            #with record_function("### backward ###"):
            loss.backward()
            #with record_function("### optimizer ###"):
            self.optimizer.step()

            losses.append(loss.item())
            # prof.step()

        return losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[tuple[str, str]]]:
        """
        Validate the model on the given dataloader with paired inputs

        Args:
            val_dataloader: Dataloader loading the validation data
            save_scores: Whether to save the scores
            
        Returns:
            Tuple(losses, embeddings, labels, scores, file_names)
            - losses: List of loss values
            - labels: List of ground truth labels (1 for same speaker, 0 for different)
            - scores: List of cosine distances
            - embeddings: List of embeddings
            - file_names: List of (source_path, target_path) tuples
        """
        labels = []
        scores = []  # Will store cosine distances
        embeddings = []
        file_names = []

        self.model.eval()
        with torch.no_grad():
            for (source_paths, target_paths), (source_wf, target_wf), label in tqdm(val_dataloader):
                # Move data to device
                source_wf = source_wf.to(self.device)
                target_wf = target_wf.to(self.device)
                label = label.to(self.device)

                # Get embeddings for both source and target
                source_embeddings = self.model(source_wf)
                target_embeddings = self.model(target_wf)
                
                # Compute cosine similarity
                similarities = cosine_similarity(source_embeddings, target_embeddings)

                embeddings.extend(source_embeddings.cpu().tolist())

                if save_scores:
                    file_names.extend(list(zip(source_paths, target_paths)))
            
                labels.extend(label.cpu().tolist())
                scores.extend(similarities.cpu().tolist())

        return labels, scores, embeddings, file_names