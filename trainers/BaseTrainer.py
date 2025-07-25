import torch
import os
import yaml
""" from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.models.speaker_model import get_speaker_model """

from .utils import calculate_EER

class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

    def save_model(self, path: str):
        """
        Save the model to the given path
        If model is a PyTorch model, it will be saved using torch.save(state_dict)
        Problem is when non-PyTorch model contains a Pytorch component (e.g. extractor). In that case,
        the trainer should implement custom saving/loading methods.

        param path: Path to save the model to
        """
        if isinstance(self.model, torch.nn.Module):
            os.makedirs(os.path.dirname(os.path.join("checkpoints",path)), exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join("checkpoints",path))
        else:
            raise NotImplementedError(
                "Child classes for non-PyTorch models need to implement save_model method"
            )

    def load_model(self, path: str):
        """
        Load the model from the given path
        Try to load the model as a PyTorch model using torch.load,
        otherwise, the child class trainer should implement custom loading method.

        param path: Path to load the model from
        """
        try:
            if 'ResNet' in self.model.feature_processor.__class__.__name__:
                """ config_path = os.path.join(path, 'config.yaml')
                model_path = os.path.join(path, 'model.pt')
                with open(config_path, 'r') as fin:
                    configs = yaml.load(fin, Loader=yaml.FullLoader)
                self.model = get_speaker_model(
                    configs['model'])(**configs['model_args'])
                load_checkpoint(self.model, model_path) """
            else:
                self.model.load_state_dict(torch.load(path, map_location=self.device))

        except FileNotFoundError:
            raise
        except:  # Path correct, but not a PyTorch model
            # raise NotImplementedError(
            #     "Child classes for non-PyTorch models need to implement load_model method"
            # )
            raise

    def calculate_EER(self, labels, predictions, plot_det: bool, det_subtitle: str) -> float:
        return calculate_EER(type(self.model).__name__, labels, predictions, plot_det, det_subtitle)

    def train(self, train_dataloader, val_dataloader, numepochs: int = 20):
        raise NotImplementedError("Child classes should implement the train method")
    
    def val(self, val_dataloader):
        raise NotImplementedError("Child classes should implement the val method")

    def eval(self, eval_dataloader, subtitle: str = ""):
        raise NotImplementedError("Child classes should implement the eval method")
