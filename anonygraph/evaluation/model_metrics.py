import tensorflow as tf
from tensorflow.keras import metrics

class MeanSquaredRootAbsoluteError(metrics.Metric):
    def __init__(self, name="squared_root_mean_absolute_error", **kwargs):
        super().__init__(name=name, **kwargs)

        self.errors = self.add_weight(name="msrae", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        squared_root_y_true = tf.sqrt(y_true)
        squared_root_y_pred = tf.sqrt(y_pred)

        current_errors = tf.abs(squared_root_y_pred - squared_root_y_true)
        self.errors.assign_add(tf.reduce_sum(current_errors))

    def result(self):
        return self.errors