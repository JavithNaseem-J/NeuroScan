from NeuroScan.config_entity.config_entity import DataIngestionConfig
from NeuroScan.utils.logging import logger
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
from pathlib import Path
import glob

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def set_kaggle_credentials(self):
        os.environ["KAGGLE_USERNAME"] = self.config.kaggle_username
        os.environ["KAGGLE_KEY"] = self.config.kaggle_api_key
        logger.info("Kaggle credentials set for API access.")

    def download_file(self):
        self.set_kaggle_credentials()

        if not os.path.exists(self.config.extracted_data_dir) or not os.listdir(self.config.extracted_data_dir):
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            logger.info("Downloading dataset from Kaggle API...")

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                dataset=self.config.download_url,
                path=str(self.config.raw_data_dir),
                unzip=False,
                quiet=False
            )

            zip_files = glob.glob(str(self.config.raw_data_dir / "*.zip"))
            if not zip_files:
                raise FileNotFoundError("No .zip file found in raw_data_dir after download.")

            zip_path = zip_files[0]  # assume only one zip
            logger.info(f"Dataset downloaded as: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.raw_data_dir)
            logger.info(f"Dataset extracted to: {self.config.raw_data_dir}")

            os.remove(zip_path)
        else:
            logger.info(f"Dataset already exists at: {self.config.extracted_data_dir}")
