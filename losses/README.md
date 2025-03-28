# Loss Functions Module

This module contains implementations of various loss functions that can be used for training neural network models in the ca-mhfa framework.

## Available Loss Functions

Currently, the following loss functions are implemented:

- **CrossEntropyLoss**: Standard cross-entropy loss for classification tasks
- **AdditiveAngularMarginLoss (ArcFace)**: Additive angular margin-based loss that operates on the processed embeddings from the feature processor, before they are passed to the final classifier

Future implementations may include:
- **CosFace**: Large margin cosine loss for deep face recognition
- **MagFace**: Magnitude-aware margin loss that adapts the margin based on feature norms

## Usage

To use a loss function, specify it with the `--loss` argument when running the training script:

```bash
python train_and_eval.py --local --dataset ASVspoof2019LADataset_pair --extractor WavLM_base --processor Mean --classifier FF --loss CrossEntropy
```

### Important: Using Embedding-Based Losses

When using embedding-based losses like `AdditiveAngularMargin`, you must:

1. Explicitly specify an embedding-compatible classifier (not the standard `FF`):
```bash
python train_and_eval.py --local --dataset ASVspoof2019LADataset_pair --extractor WavLM_base --processor Mean --classifier EmbeddingFF --loss AdditiveAngularMargin
```

Embedding-compatible classifiers are defined in the `EMBEDDING_COMPATIBLE_CLASSIFIERS` list in `common.py` and currently include:
- `EmbeddingFF`: Returns both classifier outputs and processed embeddings from the feature processor

These classifiers allow the loss function to operate on embeddings directly, enabling more powerful metric learning capabilities.

### AdditiveAngularMarginLoss Parameters

For the AdditiveAngularMarginLoss, you can customize its behavior with the following parameters:

- `--margin`: Angular margin value (default: 0.5)
- `--s`: Feature scale/radius (default: 30.0)
- `--easy_margin`: Flag to use easy margin version (default: False)

Example:
```bash
python train_and_eval.py --local --dataset ASVspoof2019LADataset_pair --extractor WavLM_base --processor Mean --classifier EmbeddingFF --loss AdditiveAngularMargin --margin 0.3 --s 25.0 --easy_margin
```

## Adding New Loss Functions

To add a new loss function:

1. Create a new file in the `losses` directory (e.g., `NewLoss.py`)
2. Implement your loss class inheriting from `BaseLoss`
3. Add the loss to the `__init__.py` file
4. Add the loss class to the `LOSSES` dictionary in `common.py`
5. Add loss metadata to the `LOSS_METADATA` dictionary in `common.py`
6. Add any required parameters to `parse_arguments.py`

### Adding Embedding-Based Loss Functions

For embedding-based loss functions like CosFace or MagFace:

1. Follow the steps above
2. In the `LOSS_METADATA` dictionary, set the type to "embedding"
3. Define all parameters needed for the loss in the "parameters" section
4. Ensure your loss function accepts processed embeddings and labels as input
5. If needed, create a new embedding-compatible classifier and add it to `EMBEDDING_COMPATIBLE_CLASSIFIERS`
6. Document the usage and parameters in this README
7. Instruct users to specifically select an embedding-compatible classifier when using your loss function 