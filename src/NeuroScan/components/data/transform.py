import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from NeuroScan.config_entity.config_entity import DataTransformationConfig
from NeuroScan.utils.helpers import create_directories, read_yaml
from NeuroScan.utils.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):

        self.config = config
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.cropped_train_dir = self.config.source_data_dir / 'Crop-Brain-MRI'
        self.cropped_test_dir = self.config.source_data_dir / 'Test-Data'

    def save_preprocessed_data(self, train_dir, val_dir, test_dir):

        try:
            train_datagen = ImageDataGenerator(
                rotation_range=10,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            test_datagen = ImageDataGenerator()
            train_gen = train_datagen.flow_from_directory(
                directory=str(train_dir),
                target_size=(240, 240),
                batch_size=self.config.batch_size,
                class_mode=self.config.class_mode,
                subset='training',
                shuffle=False
            )
            val_gen = train_datagen.flow_from_directory(
                directory=str(train_dir),
                target_size=(240, 240),
                batch_size=self.config.batch_size,
                class_mode=self.config.class_mode,
                subset='validation',
                shuffle=False
            )
            test_gen = test_datagen.flow_from_directory(
                directory=str(test_dir),
                target_size=(240, 240),
                batch_size=self.config.batch_size,
                class_mode=self.config.class_mode,
                shuffle=False
            )

            train_data = []
            train_labels = []
            val_data = []
            val_labels = []
            test_data = []
            test_labels = []
            for _ in range(train_gen.n // train_gen.batch_size + 1):
                x, y = next(train_gen)
                train_data.append(x)
                train_labels.append(y)
            for _ in range(val_gen.n // val_gen.batch_size + 1):
                x, y = next(val_gen)
                val_data.append(x)
                val_labels.append(y)
            for _ in range(test_gen.n // test_gen.batch_size + 1):
                x, y = next(test_gen)
                test_data.append(x)
                test_labels.append(y)

            train_data = np.concatenate(train_data, axis=0)
            train_labels = np.concatenate(train_labels, axis=0)
            val_data = np.concatenate(val_data, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)
            test_data = np.concatenate(test_data, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            np.save(self.config.saved_train_gen_path, {'data': train_data, 'labels': train_labels})
            np.save(self.config.saved_val_gen_path, {'data': val_data, 'labels': val_labels})
            np.save(self.config.saved_test_gen_path, {'data': test_data, 'labels': test_labels})
            logger.info(f"Preprocessed data saved to {self.config.saved_train_gen_path}, {self.config.saved_val_gen_path}, {self.config.saved_test_gen_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise

    def load_preprocessed_data(self):
        """Loads preprocessed data if available."""
        try:
            if (os.path.exists(self.config.saved_train_gen_path) and
                os.path.exists(self.config.saved_val_gen_path) and
                os.path.exists(self.config.saved_test_gen_path)):
                train_data = np.load(self.config.saved_train_gen_path, allow_pickle=True).item()
                val_data = np.load(self.config.saved_val_gen_path, allow_pickle=True).item()
                test_data = np.load(self.config.saved_test_gen_path, allow_pickle=True).item()
                logger.info("Preprocessed data loaded from saved files.")
                return train_data, val_data, test_data

            logger.warning("No saved preprocessed data found; transformation required.")
            return None, None, None

        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise

    def create_data_generators(self, train_data=None, val_data=None, test_data=None):
        """Creates data generators from preprocessed data or directories."""

        try:
            if train_data is None or val_data is None or test_data is None:
                logger.info("Creating generators from directories...")
                train_datagen = ImageDataGenerator(
                    rotation_range=10,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    validation_split=0.2
                )
                test_datagen = ImageDataGenerator()
                train_generator = train_datagen.flow_from_directory(
                    directory=str(self.cropped_train_dir),
                    target_size=(240, 240),
                    batch_size=self.config.batch_size,
                    class_mode=self.config.class_mode,
                    subset='training'
                )
                valid_generator = train_datagen.flow_from_directory(
                    directory=str(self.cropped_train_dir),
                    target_size=(240, 240),
                    batch_size=self.config.batch_size,
                    class_mode=self.config.class_mode,
                    subset='validation'
                )
                test_generator = test_datagen.flow_from_directory(
                    directory=str(self.cropped_test_dir),
                    target_size=(240, 240),
                    batch_size=self.config.batch_size,
                    class_mode=self.config.class_mode,
                    shuffle=False
                )

                # Validate cardinality for directory-based generators
                if (train_generator.n == 0 or valid_generator.n == 0 or test_generator.n == 0):
                    logger.error("One or more data generators are empty.")
                    raise ValueError("Data generators contain no images.")
                logger.info(f"Data generators created: {train_generator.n} training, "
                            f"{valid_generator.n} validation, {test_generator.n} test images.")
            else:
                logger.info("Creating generators from preprocessed data...")
                train_generator = tf.data.Dataset.from_tensor_slices((train_data['data'], train_data['labels'])).batch(self.config.batch_size)
                valid_generator = tf.data.Dataset.from_tensor_slices((val_data['data'], val_data['labels'])).batch(self.config.batch_size)
                test_generator = tf.data.Dataset.from_tensor_slices((test_data['data'], test_data['labels'])).batch(self.config.batch_size)

                if (train_generator.cardinality().numpy() == 0 or valid_generator.cardinality().numpy() == 0 or test_generator.cardinality().numpy() == 0):
                    logger.error("One or more data generators are empty.")
                    raise ValueError("Data generators contain no images.")
                
                logger.info(f"Data generators created: {train_generator.cardinality().numpy() * self.config.batch_size} training, "
                            f"{valid_generator.cardinality().numpy() * self.config.batch_size} validation, "
                            f"{test_generator.cardinality().numpy() * self.config.batch_size} test images.")

            return train_generator, valid_generator, test_generator

        except Exception as e:
            logger.error(f"Error creating data generators: {e}")
            raise

    def transform(self):
        try:
            logger.info("Starting data transformation pipeline...")
            train_data, val_data, test_data = self.load_preprocessed_data()
            if train_data is None:
                self.save_preprocessed_data(self.cropped_train_dir, self.cropped_train_dir, self.cropped_test_dir)
                train_data, val_data, test_data = self.load_preprocessed_data()
            train_gen, valid_gen, test_gen = self.create_data_generators(train_data, val_data, test_data)
            logger.info("Data transformation pipeline completed.")
            return train_gen, valid_gen, test_gen

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise