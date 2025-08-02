from NeuroScan.components.model.training import ModelTraining
from NeuroScan.components.model.evaluation import ModelEvaluator
from NeuroScan.config.model_config import ModelConfigurationManager
from NeuroScan.utils.logging import logger
from NeuroScan.utils.exceptions import CustomException


class ModelPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ModelConfigurationManager()
        training_config = config.get_model_training_config()
        model_trainer = ModelTraining(config=training_config)
        model_trainer.train_model()

        evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluator(config=evaluation_config)
        model_evaluator.evaluate_model()