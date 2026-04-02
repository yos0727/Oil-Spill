# Oil Spill Project

---

## 📁 Key Project Files
- [create_dataset.py](#-create_datasetpy)  
- [trainer.py](#-trainerpy)  

---

## ⚙️ Environment Requirements

- Python >= 3.10
- PyTorch
- torchvision
- NumPy
- OpenCV (opencv-python)
- scikit-learn
- Ultralytics
- PyYAML
>⚠️ It's a incomplete requirements list

---

## 📄 create_dataset.py
### Class `DatasetBuilder`
`DatasetBuilder` is a utility class for automatically constructing dataset splits from a directory-based image/label dataset.
```python
class DatasetBuilder:
    def __init__(self,
        src: str,
        dst: str,
        img_ext: Iterable[str],
        lbl_ext: Iterable[str],
        override: bool = False,
        random_seed: int = None
    ) -> None:
```
* `src` : Path to the source dataset directory
* `dst` : Output directory for generated split files
* `img_ext` : Valid image extensions (e.g. [".jpg", ".png"])
* `lbl_ext` : Valid label extensions (e.g. [".png"])
* `override` : Whether to overwrite an existing output directory
* `random_seed` : Random seed for reproducible splits

> ⚠️ **DatasetBuilder Limitations and Constraints**
> 1. Image and label paths **must be mutually mappable** by replacing `/images/` with `/labels/` in the directory structure.
> 2. Image file extensions and label file extensions **must not overlap** (e.g., image extensions should not include `.png` if labels also use `.png`).

### Create Dataset Examples
```python
builder = DatasetBuilder(
    src="dataset",
    dst="splits",
    img_ext=[".jpg", ".jpeg"],
    lbl_ext=[".png"],
    override=True,
    random_seed=42
)
```

### Example 1: Train / Val / Test Split
```python
builder.create(ratio=(6, 2, 2)) # ratio = (train, val, test)
```
### Output
```
splits/
├── train.txt
├── val.txt
└── test.txt
```

### Example 2: Train / Val Split Only
```python
builder.create(ratio=(8, 2))
```
### Output
```
splits/
├── train.txt
└── val.txt
```

### Example 3: K-Fold
```python
builder.kfold(k=5)
```
### Output
```
splits/
├── fold1/
│   ├── train.txt
│   └── val.txt
├── fold2/
│   ├── train.txt
│   └── val.txt
└── ...
```

---

## 📄 trainer.py

### Class `BaseTrainer`
`BaseTrainer` is an **abstract base class (ABC)** responsible for:

* Control the overall workflow (training / inference / evaluation)
* Standardizing metric computation and output formats

### Basic Usage
1. Load or initialize the model
    ```python
    trainer.load(weights=r"path\to\weights.pt")
    ```
   * `weights`: Path to a pretrained weights file. If None, the model is initialized with random weights.

2. Train the model
    ```python
    trainer.train(src="dataset", dst="output", save=True, epochs=10, batch_size=8)
    ```
   * `src`: Path to dataset folder
   * `dst`: Directory to save weights (required if save=True)
   * `save`: Whether to save model weights
   * `**kwargs`: Training configuration (epochs, batch_size, learning_rate, etc.) varies between models

3. Run inference
    ```python
    trainer.test(src="dataset", dst="output", file_name="test.txt", color_coded=True, log=True, save=True)
    ```
   * `src`: Path to dataset folder
   * `dst`: Directory to save inference result (required if color_coded or log or save is True)
   * `file_name`: Name of the file listing the paths of images to predict
   * `color_coded`: Whether to save the color coded mask
   * `log`: Whether output the metrics
   * `save`: Whether to save the metrics

4. K-Fold cross-validation
    ```python
    trainer.kfold(src="kfold_dataset", dst="output", save=True, epochs=10, batch_size=8)
    ```
   * `src`: Path to kfold dataset folder
   * `dst`: Directory to save weights (required if save=True)
   * `save`: Whether to save the metrics
   * `**kwargs`: Training configuration (epochs, batch_size, learning_rate, etc.) varies between models

### Usage Example
#### Example 1: Train & Test
```python
src = r"datasets\dataset_42"
dst = r"runs\model_name"

trainer = ModelTrainer() # ModelTrainer is a subclass inherit from BaseTrainer
trainer.load_model()
trainer.train(src, epochs=10, batch=16, save=False)
trainer.test(src, dst, save=True)
```
#### Example 2: Cross Validation
```python
src = r"datasets\dataset_42"
dst = r"runs\model_name"

trainer = ModelTrainer() # ModelTrainer is a subclass inherit from BaseTrainer
trainer.load_model()
trainer.kfold(src, epochs=10, batch=16, save=True)
```

### Concrete New Model
only need to inherit from `BaseTrainer` and implement the following abstract methods:
```python
def load_model(self, 
    weights: str | None = None
) -> None:
    """
    Initialize the model and optionally load pretrained weights.

    Parameters
    ----------
    weights : Path to a pretrained weights file. If None, the model is initialized with random weights.

    Notes
    -----
    This method should assign the model instance to `self.model`.
    No training or forward pass should occur here; this is purely model initialization.
    """
```

```python
def _train(
    self, 
    src: str,
    dst: str | None = None, 
    save: bool = True, 
    **kwargs
) -> None:
    """
    Execute the complete training procedure for the model.

    Parameters
    ----------
    src : Path to the directory containing the dataset for training.
    dst : Directory where the trained model weights will be saved. Required if `save=True`.
    save : Whether to save the trained model after training.
    **kwargs : Additional keyword arguments for training configuration, such as: epochs, image_size

    Notes
    -----
    This method will be called by other and the path checking already implements in the caller method
    """
```

```python
def _predict(
    self, 
    src: str,
    file_name: str = "test.txt"
) -> None:
    """
    Run inference using the trained model on a given dataset.

    Parameters
    ----------
    src : Path to the directory containing the dataset for inference.
    file_name : Name of the file listing the paths of images to predict.

    Notes
    -----
    Results must be stored in self.results as tuples (file_name, pred_mask, gt_mask):
        file_name (str): the name of the image.
        pred_mask (ndarray): the model predicted mask
        gt_mask (ndarray): the ground truth binary mask
    Example:
        new_prediction = (file_name, pred_mask, gt_mask)
        self.results.append(new_prediction)
    This method will be called by other and the path checking already implements in the caller method
    """
```