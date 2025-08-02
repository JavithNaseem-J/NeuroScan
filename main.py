import os
import sys
import argparse
from NeuroScan.pipelines.feature_pipline import FeaturePipeline
from NeuroScan.pipelines.model_pipeline import ModelPipeline
from NeuroScan.utils.logging import logger
from NeuroScan.utils.exceptions import CustomException

def run_stages(stage: str):
    logger.info(f"Starting stage: {stage}")
    try:
        if stage == "feature_pipeline":
            logger.info("Running Feature Pipeline...")
            feature_pipeline = FeaturePipeline()
            feature_pipeline.run()
        elif stage == "model_pipeline":
            logger.info("Running Model Pipeline...")
            model_pipeline = ModelPipeline()
            model_pipeline.run()
        else:
            raise ValueError(f"Unknown stage: {stage}")
        logger.info(f"Stage {stage} completed successfully.")

    except CustomException as e:
        logger.error(f"Error occurred while running stages: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeuroScan Pipelines")
    parser.add_argument("--stage", help="Stage to run: feature_pipeline or model_pipeline", required=True)
    args = parser.parse_args()

    if args.stage:
        run_stages(args.stage)
    
    else:
        stages = [
            "feature_pipeline",
            "model_pipeline"
        ]
