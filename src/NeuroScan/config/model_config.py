from pathlib import Path
from NeuroScan.utils.helpers import read_yaml, create_directories
from NeuroScan.config_entity.config_entity import ModelTrainerConfig
from NeuroScan.config_entity.config_entity import ModelEvaluationConfig
from NeuroScan.utils.logging import logger
from NeuroScan.constants.paths import CONFIG_PATH, PARAMS_PATH

class ModelConfigurationManager:
    
    def __init__(self, config_file=CONFIG_PATH, params_file=PARAMS_PATH):
        self.config = read_yaml(config_file)
        self.params = read_yaml(params_file)
        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_params
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_ckpt_path=config.model_ckpt_path,
            train_gen_path=config.train_gen_path,
            val_gen_path=config.val_gen_path,
            mlflow_uri=config.mlflow_uri,
            experiment_name=config.experiment_name,
            input_shape=params.input_shape,
            num_classes=params.num_classes,
            base_model_name=params.base_model_name,
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            dropout_rate=params.dropout_rate,
            loss=params.loss,
            monitor=params.monitor,
            patience=params.patience,
            reduce_lr_factor=params.reduce_lr_factor,
            reduce_lr_patience=params.reduce_lr_patience,
            reduce_lr_min_delta=params.reduce_lr_min_delta
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.evaluation
        params = self.params.transform
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_gen_path=Path(config.test_gen_path),
            metrics_file_path=config.metrics_file_path,
            model_file=Path(config.model_file),
            confusion_matrix_plot_path=config.confusion_matrix_plot_path,
            mlflow_uri=config.mlflow_uri,
            experiment_name=config.experiment_name,
            batch_size=params.batch_size,
            target_image_size=params.target_image_size
        )