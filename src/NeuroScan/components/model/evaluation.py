import numpy as np
import json
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt
from NeuroScan.utils.logging import logger
from NeuroScan.config_entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score,ConfusionMatrixDisplay

class ModelEvaluator:


    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = tf.keras.models.load_model(self.config.model_file)
        self.test_generator = None
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        self.run = mlflow.start_run()
        self._initialize_test_generator()


    def _initialize_test_generator(self):

        try:
            test_data = np.load(self.config.test_gen_path, allow_pickle=True).item()
            if test_data['data'].shape[1:3] != tuple(self.config.target_image_size):
                logger.warning(f"Resizing test data from {test_data['data'].shape[1:3]} to {self.config.target_image_size}")
                test_data['data'] = np.array([tf.image.resize(img, self.config.target_image_size).numpy() for img in test_data['data']])
            self.test_generator = tf.data.Dataset.from_tensor_slices((test_data['data'], test_data['labels'])).batch(self.config.batch_size)

            logger.info(f"Test generator initialized with {self.test_generator.cardinality().numpy() * self.config.batch_size} samples.")
        except Exception as e:
            logger.error(f"Error initializing test generator: {e}")
            raise

    def evaluate(self):
        """Evaluates the model on the test data, saves metrics and plot."""
        try:
            logger.info("Evaluating model on test data...")
            test_steps = self.test_generator.cardinality().numpy()
            y_true = []
            y_pred = []
            for x_batch, y_batch in self.test_generator.take(test_steps):
                y_pred_batch = self.model.predict(x_batch)
                y_true.extend(np.argmax(y_batch, axis=1))
                y_pred.extend(np.argmax(y_pred_batch, axis=1))

            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            f1 = f1_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)

            metrics = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "f1_score": float(f1),
                "recall": float(recall),
                "precision": float(precision)
            }

            with open(self.config.metrics_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.config.metrics_file_path}")

            confusion_matrix_plot_path = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['glioma', 'meningioma', 'notumor', 'pituitary'])
            confusion_matrix_plot_path.plot(cmap='Blues', values_format='d')
            plt.title('Confusion Matrix')
            plt.savefig(self.config.confusion_matrix_plot_path)
            plt.close()
            logger.info(f"Confusion matrix plot saved to {self.config.confusion_matrix_plot_path}")

            # Log metrics and artifacts to MLflow
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_artifact(self.config.metrics_file_path)
            mlflow.log_artifact(self.config.confusion_matrix_plot_path)
            mlflow.log_text(str(cm), "confusion_matrix.txt")

            logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, "
                        f"F1-Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
            logger.info(f"Confusion Matrix:\n{cm}")
            return test_loss, test_accuracy, cm, report
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
        finally:
            mlflow.end_run()