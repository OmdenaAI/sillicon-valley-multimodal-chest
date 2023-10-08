
import os
import sys
from srcp.logger import logging
from srcp.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()