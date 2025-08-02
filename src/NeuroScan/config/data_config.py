import os
from pathlib import Path
from dotenv import load_dotenv
from NeuroScan.utils.helpers import read_yaml, create_directories
from NeuroScan.config_entity.config_entity import DataIngestionConfig
from NeuroScan.config_entity.config_entity import DataCleaningConfig
from NeuroScan.config_entity.config_entity import DataTransformationConfig
from NeuroScan.utils.logging import logger
from NeuroScan.constants.paths import CONFIG_PATH, PARAMS_PATH


class DataConfigurationManager:
    def __init__(self, config_file=CONFIG_PATH, params_file=PARAMS_PATH):
        self.config = read_yaml(config_file)
        self.params = read_yaml(params_file)

        load_dotenv()

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.ingestion

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            download_url=config.download_url,
            raw_data_dir=Path(config.raw_data_dir),
            extracted_data_dir=Path(config.extracted_data_dir),
            kaggle_username=os.getenv("kaggle_username"),
            kaggle_api_key=os.getenv("kaggle_api_key")
        )
    

    def get_data_cleaning_config(self):
        config = self.config.cleaning
        params = self.params.cleaning
        create_directories([config.root_dir])
        return DataCleaningConfig(
            root_dir=Path(config.root_dir),
            source_data_dir=Path(config.source_data_dir),
            image_size=params.image_size,
            target_image_size=params.target_image_size
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.transformation
        params = self.params.transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            source_data_dir=Path(config.source_data_dir),
            batch_size=params.batch_size,
            class_mode=config.class_mode,
            saved_train_gen_path=config.saved_train_gen_path,
            saved_val_gen_path=config.saved_val_gen_path,
            saved_test_gen_path=config.saved_test_gen_path
        )