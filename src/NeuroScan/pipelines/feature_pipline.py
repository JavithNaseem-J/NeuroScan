from logging import config
from NeuroScan.components.data.ingestion import DataIngestion
from NeuroScan.components.data.cleaning import DataCleaning
from NeuroScan.components.data.transform import DataTransformation
from NeuroScan.config.data_config import DataConfigurationManager
from NeuroScan.utils.logging import logger
from NeuroScan.utils.exceptions import CustomException


class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):
        config = DataConfigurationManager()


        ingestion_config = config.get_data_ingestion_config()
        data_ingestor = DataIngestion(config=ingestion_config)
        data_ingestor.download_file()

        cleaning_config = config.get_data_cleaning_config()
        data_cleaner = DataCleaning(config=cleaning_config)
        data_cleaner.clean_data()

        transformation_config = config.get_data_transformation_config()
        data_transformer = DataTransformation(config=transformation_config)
        data_transformer.transform_data()