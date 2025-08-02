from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    download_url: str
    raw_data_dir: Path
    extracted_data_dir: Path
    kaggle_username: str
    kaggle_api_key: str

@dataclass
class DataCleaningConfig:
    root_dir: Path
    source_data_dir: Path
    image_size: int
    target_image_size: list


@dataclass
class DataTransformationConfig:
    root_dir: Path
    source_data_dir: Path
    batch_size: int
    class_mode: str
    saved_train_gen_path: str
    saved_val_gen_path: str
    saved_test_gen_path: str


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_path: str
    train_gen_path: str
    val_gen_path: str
    mlflow_uri: str
    experiment_name: str
    input_shape: list
    num_classes: int
    learning_rate: float
    epochs: int
    dropout_rate: float
    loss: str
    monitor: str
    patience: int
    reduce_lr_factor: float
    reduce_lr_patience: int
    reduce_lr_min_delta: float



@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_gen_path: Path
    metrics_file_path: str
    confusion_matrix_plot_path: str
    mlflow_uri: str
    experiment_name: str
    batch_size: int
    model_file: Path
    target_image_size: list