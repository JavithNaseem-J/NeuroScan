import os
import cv2
import numpy as np
from tqdm import tqdm
import imutils
from pathlib import Path
from NeuroScan.config_entity.config_entity import DataCleaningConfig
from NeuroScan.utils.helpers import create_directories
from NeuroScan.utils.logging import logger

class DataCleaning:
    """Handles image preprocessing, including cropping and resizing."""
    def __init__(self, config: DataCleaningConfig):
        self.config = config
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.train_dir = self.config.source_data_dir / 'Training'
        self.test_dir = self.config.source_data_dir / 'Testing'
        self.cropped_train_dir = self.config.root_dir / 'Crop-Brain-MRI'
        self.cropped_test_dir = self.config.root_dir / 'Test-Data'

    def validate_directories(self):
        """Validates that source directories exist."""
        try:
            for directory in [self.train_dir, self.test_dir]:
                if not directory.exists():
                    logger.error(f"Source directory {directory} does not exist.")
                    raise FileNotFoundError(f"Source directory {directory} not found.")
                for class_name in self.classes:
                    class_dir = directory / class_name
                    if not class_dir.exists():
                        logger.error(f"Class directory {class_dir} does not exist.")
                        raise FileNotFoundError(f"Class directory {class_dir} not found.")
            logger.info("All source directories validated successfully.")
        except Exception as e:
            logger.error(f"Directory validation failed: {e}")
            raise

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crops image to focus on the region of interest."""
        try:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
            img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
            img_thresh = cv2.erode(img_thresh, None, iterations=2)
            img_thresh = cv2.dilate(img_thresh, None, iterations=2)
            contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)
            if not contours:
                logger.warning("No contours found in image; returning original.")
                return image
            c = max(contours, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()])[0]
            extRight = tuple(c[c[:, :, 0].argmax()])[0]
            extTop = tuple(c[c[:, :, 1].argmin()])[0]
            extBottom = tuple(c[c[:, :, 1].argmax()])[0]
            cropped_img = image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]
            return cropped_img
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            return image

    def create_output_directories(self):
        """Creates directories for cropped training and testing images."""
        try:
            create_directories([self.cropped_train_dir, self.cropped_test_dir])
            for class_name in self.classes:
                create_directories([
                    self.cropped_train_dir / class_name,
                    self.cropped_test_dir / class_name
                ])
            logger.info(f"Output directories created at {self.cropped_train_dir} and {self.cropped_test_dir}")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise

    def process_images(self, source_dir: Path, target_dir: Path):
        """Processes images by cropping and resizing, then saves to target directory."""
        try:
            image_count = 0
            for class_name in self.classes:
                class_source = source_dir / class_name
                class_target = target_dir / class_name
                if not class_source.exists():
                    logger.warning(f"Source directory {class_source} does not exist.")
                    continue
                j = 0
                for img_name in tqdm(os.listdir(class_source), desc=f"Processing {class_name}"):
                    img_path = class_source / img_name
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Failed to read image {img_path}")
                        continue
                    cropped_img = self.crop_image(img)
                    if cropped_img is not None:
                        resized_img = cv2.resize(cropped_img, (240, 240))
                        save_path = class_target / f"{j}.jpg"
                        cv2.imwrite(str(save_path), resized_img)
                        j += 1
                        image_count += 1
                logger.info(f"Processed {j} images for class {class_name}")
            if image_count == 0:
                logger.error("No images were processed; check source directories.")
                raise ValueError("No images processed during cleaning.")
            logger.info(f"Total images processed: {image_count}")
        except Exception as e:
            logger.error(f"Error processing images: {e}")
            raise

    def clean(self):
        """Executes the full cleaning pipeline."""
        try:
            logger.info("Starting data cleaning pipeline...")
            self.validate_directories()
            self.create_output_directories()
            self.process_images(self.train_dir, self.cropped_train_dir)
            self.process_images(self.test_dir, self.cropped_test_dir)
            logger.info("Data cleaning pipeline completed.")
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise