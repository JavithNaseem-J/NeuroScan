import os
import mlflow
import tensorflow as tf
import mlflow.tensorflow
import numpy as np
from NeuroScan.config_entity.config_entity import ModelTrainerConfig
from NeuroScan.utils.helpers import read_yaml, create_directories
from NeuroScan.utils.logging import logger
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None
        self.history = None
        self.train_generator = None
        self.valid_generator = None

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)

        self.run = mlflow.start_run()
        self._initialize_generators()

    def _initialize_generators(self):
        try:
            if os.path.exists(self.config.train_gen_path) and os.path.exists(self.config.val_gen_path):
                logger.info("Loading pre-saved generators...")
                train_data = np.load(self.config.train_gen_path, allow_pickle=True).item()
                val_data = np.load(self.config.val_gen_path, allow_pickle=True).item()
                self.train_generator = tf.data.Dataset.from_tensor_slices((train_data['data'], train_data['labels'])).batch(32)  
                self.valid_generator = tf.data.Dataset.from_tensor_slices((val_data['data'], val_data['labels'])).batch(32) 
                logger.info("Pre-saved generators loaded successfully.")
            else:
                logger.error("Saved generator files not found. Please run data_transformation first.")
                raise FileNotFoundError("Generator files missing.")
        except Exception as e:
            logger.error(f"Error initializing generators: {e}")
            raise

    def build_model(self):
        """Constructs the model architecture."""

        try:
            base_model = EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_shape=tuple(self.config.input_shape)
            )
            model = base_model.output
            model = GlobalMaxPooling2D()(model)
            model = Dropout(self.config.dropout_rate)(model)
            model = Dense(self.config.num_classes, activation="softmax")(model)
            self.model = Model(inputs=base_model.input, outputs=model)
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss=self.config.loss,
                metrics=['accuracy']
            )
            logger.info("Model architecture built and compiled successfully.")

        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def setup_callbacks(self):

        try:
            checkpoint = ModelCheckpoint(
                self.config.model_path,
                monitor=self.config.monitor,
                save_best_only=True,
                mode='auto',
                verbose=1
            )
            earlystop = EarlyStopping(
                monitor=self.config.monitor,
                patience=self.config.patience,
                mode='auto',
                verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor=self.config.monitor,
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_delta=self.config.reduce_lr_min_delta,
                mode='auto',
                verbose=1
            )
            return [checkpoint, earlystop, reduce_lr]

        except Exception as e:
            logger.error(f"Error setting up callbacks: {e}")
            raise

    def log_training_metrics(self, history):
        
        try:
            train_acc = np.array(history.history['accuracy'])
            val_acc = np.array(history.history['val_accuracy'])
            mean_train_acc = np.mean(train_acc)
            std_train_acc = np.std(train_acc)
            mean_val_acc = np.mean(val_acc)
            std_val_acc = np.std(val_acc)
            mlflow.log_metric("mean_train_accuracy", mean_train_acc)
            mlflow.log_metric("std_train_accuracy", std_train_acc)
            mlflow.log_metric("mean_val_accuracy", mean_val_acc)
            mlflow.log_metric("std_val_accuracy", std_val_acc)
            logger.info(f"Logged training metrics: mean_train_acc={mean_train_acc:.4f}, std_train_acc={std_train_acc:.4f}, "
                        f"mean_val_acc={mean_val_acc:.4f}, std_val_acc={std_val_acc:.4f}")

        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")
            raise

    def train(self):
        try:
            if self.model is None:
                self.build_model()

            callbacks = self.setup_callbacks()
            logger.info("Starting model training...")
            self.history = self.model.fit(
                self.train_generator,
                epochs=self.config.epochs,
                validation_data=self.valid_generator,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Model training completed.")
            self.log_training_metrics(self.history)
            mlflow.tensorflow.log_model(self.model, "model")
            return self.model, self.history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

        finally:
            mlflow.end_run()
