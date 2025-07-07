import math
import os
import torch
from torch.nn import CrossEntropyLoss
from matplotlib import pyplot as plt
import numpy as np

from trainers.BaseTrainer import BaseTrainer
from losses.CrossEntropyLoss import CrossEntropyLoss as CustomCrossEntropyLoss

# This is the base class for all trainers - use this if you aren't experimenting with new approaches
class BaseFFTrainer(BaseTrainer):
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", save_embeddings=False):
        super().__init__(model, device)

        self.optimizer = torch.optim.Adam(
            model.parameters()
        )  # Can play with lr and weight_decay for regularization
        self.device = device

        self.model = model.to(device)
        self.save_embeddings = save_embeddings
        if save_embeddings:
            os.makedirs("embeddings", exist_ok=True)
        
        # Move loss function to the correct device if it has parameters
        #if hasattr(self.model.loss_fn, 'to') and callable(getattr(self.model.loss_fn, 'to')):
        #    self.model.loss_fn = self.model.loss_fn.to(device)

        # A statistics tracker dict for the training and validation losses, accuracies and EERs
        self.statistics = {
            "train_losses": [],
            "val_eers": [],
        }

    def train(self, train_dataloader, val_dataloader, numepochs=20, start_epoch=0):
        """
        Common training loop

        Train the model on the given dataloader for the given number of epochs
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to train for
        param start_epoch: Epoch to start from (1-indexed)
        """
        self.model.train()
        for epoch in range(start_epoch+1, start_epoch + numepochs + 1):  # 1-indexed epochs
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            #self.model.train()  # Set model to training mode

            losses = self.train_epoch(train_dataloader)

            # Save epoch statistics
            epoch_loss = np.mean(losses)
            print(
                f"Epoch {epoch} finished,",
                f"training loss: {epoch_loss}",
            )

            self.statistics["train_losses"].append(epoch_loss)

            # Every epoch
            # plot losses and accuracy and save the model
            # validate on the validation set (incl. computing EER)
            self._plot_loss(
                self.statistics["train_losses"],
                f"Training epoch {epoch}",
            )
            self.save_model(f"./{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_{epoch}.pt")

            # Validation
            epochs_to_val = 1
            if epoch % epochs_to_val == 0:
                eer = self.val(val_dataloader)
                print(f"Validation EER: " + "None" if eer == None else f"{eer*100}%")
                self.statistics["val_eers"].append(eer)

        self._plot_eer(self.statistics["val_eers"], "Validation")

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model for one epoch on the given dataloader

        return: Tuple(list of accuracies, list of losses)
        """
        raise NotImplementedError("Child classes should implement the train_epoch method")

    def val(
        self, val_dataloader, save_scores=False, plot_det=False, subtitle=""
    ) -> tuple[float, float, float | None]:
        """
        Common validation loop

        Validate the model on the given dataloader and return the loss, accuracy and EER

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(loss, accuracy, EER)
        """
        self.model.eval()
        with torch.no_grad():
            labels, scores, predictions, file_names = self.val_epoch(val_dataloader, save_scores)

            if save_scores:
                os.makedirs("scores", exist_ok=True)
                with open(f"scores/{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_{subtitle}_scores.txt", "w") as f:
                    for file_name, score, label in zip(file_names, scores, labels):
                        f.write(f"{file_name},{score},{'nan' if math.isnan(label) else int(label)}\n")

            if self.save_embeddings and predictions:
                embeddings = np.array(predictions)
                embeddings_file = f"embeddings/{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_{subtitle}_embeddings.npz"
                np.savez_compressed(
                    embeddings_file,
                    embeddings=embeddings,
                    file_names=file_names
                )
                print(f"Saved embeddings to {embeddings_file}")
            # Skip EER calculation if any of the labels is None or all labels are the same
            if None in labels or any(map(lambda x: math.isnan(x), labels)):
                print("Skipping EER calculation due to missing labels")
                eer = None
            elif len(set(labels)) == 1:
                print("Skipping EER calculation due to all labels being the same")
                eer = None
            else:
                eer = self.calculate_EER(labels, scores, plot_det=plot_det, det_subtitle=subtitle)

            return eer

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        """
        Validate the model for one epoch on the given dataloader

        return: Tuple(list of losses, list of labels, list of scores, list of predictions, list of file names)
        """
        raise NotImplementedError("Child classes should implement the val_epoch method")

    def eval(self, eval_dataloader, subtitle: str = ""):
        """
        Common evaluation code

        Evaluate the model on the given dataloader and print the loss, accuracy and EER

        param eval_dataloader: Dataloader loading the test data
        """

        # Reuse code from val() to evaluate the model on the eval set
        eer = self.val(
            eval_dataloader, save_scores=True, plot_det=True, subtitle=subtitle
        )
        print(f"Eval EER: {eer*100 if eer else None}%")

    def _plot_loss(self, losses, subtitle: str = ""):
        """
        Plot the loss and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss")
        plt.legend()
        plt.title(f"{type(self.model.extractor).__name__} {type(self.model.feature_processor).__name__} Loss" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"./{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_loss_{subtitle}.png")

    def _plot_eer(self, eers, subtitle: str = ""):
        """
        Plot the EER and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(eers, label="EER")
        plt.legend()
        plt.title(f"{type(self.model.extractor).__name__} {type(self.model.feature_processor).__name__} EER" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("EER")
        plt.savefig(f"./{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_EER_{subtitle}.png")

    def finetune(self, train_dataloader, val_dataloader, numepochs=5, finetune_ssl=False, start_epoch=1):
        """
        Fine-tune the model on the given dataloader for the given number of epochs.
        TODO: Maybe do finetuning based on steps instead of epochs?

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to fine-tune for
        param finetune_ssl: Whether to fine-tune the SSL extractor
        """

        self.model.train()
        self.model.extractor.finetune = finetune_ssl
        # Use the optimizer but with a smaller learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

        for epoch in range(start_epoch+1, start_epoch + numepochs + 1):
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            losses = self.train_epoch(train_dataloader)

            # Save epoch statistics
            epoch_loss = np.mean(losses)
            print(
                f"Finetuning epoch {epoch} finished,",
                f"Finetuning training loss: {epoch_loss}",
            )

            self.statistics["train_losses"].append(epoch_loss)

            self.save_model(f"./{type(self.model.extractor).__name__}_{type(self.model.feature_processor).__name__}_finetune_{epoch}.pt")

            epochs_to_val = 1  # Validate every epoch
            if epoch % epochs_to_val == 0:
                eer = self.val(val_dataloader)
                print(f"Validation EER: " + "None" if eer == None else f"{eer*100}%")
                self.statistics["val_eers"].append(eer)

        self._plot_eer(self.statistics["val_eers"], "Finetuning EER")
