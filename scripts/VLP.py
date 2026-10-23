import keras
from keras.src.utils import io_utils
import tensorflow as tf


@keras.utils.register_keras_serializable(package='Custom', name='ProjectionLayer')
class ProjectionLayer(keras.layers.Layer):
    def __init__(self, projection_var=None, output_channels=None, **kwargs):
        super(ProjectionLayer, self).__init__(**kwargs)
        self.projection_var = projection_var
        self.output_channels = output_channels
        self.weight = None

    def build(self, input_shape):
        channel_sz = input_shape[-1]
        spatial_sz = input_shape[1] * input_shape[2]

        if channel_sz < spatial_sz:
            out_ch_lim = channel_sz
        else:
            out_ch_lim = spatial_sz

        # For when lambda is smaller than or equal to 1
        if self.projection_var <= 1:
            # Initialize projection tensor at zero mean unit std dev distribution
            projection_tensor = tf.random.normal(shape=(channel_sz, out_ch_lim), mean=0, stddev=1, seed=channel_sz)

            # Obtain column indices of projection tensor and compute the square root of the column indices
            col_indices = tf.cast(tf.range(1, out_ch_lim + 1), dtype=tf.float32)
            col_sqrt = tf.reshape(tf.sqrt(col_indices), [1, out_ch_lim])

            # Create exponential decrement on projection tensor
            projection_tensor = projection_tensor / col_sqrt

            # Shuffle the preprocessed projection tensor
            col_indices = tf.range(out_ch_lim)
            shuffled_indices = tf.random.shuffle(col_indices, seed=channel_sz)
            projection_tensor = tf.gather(projection_tensor, shuffled_indices, axis=1)

            # Selection of shuffled projection tensor based on lambda
            col_var = tf.math.reduce_variance(projection_tensor, axis=0)
            exp_var = col_var / tf.reduce_sum(col_var)
            cum_sum = tf.cumsum(exp_var, axis=0)
            sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var))

            while tf.size(sel_indices).numpy() == 0:
                # Condition 1: First cumulative sum of explained variance already larger than the desired sum
                if tf.math.greater(cum_sum[0], self.projection_var) and tf.less_equal(tf.reduce_min(exp_var),
                                                                                      self.projection_var):
                    exp_var = exp_var[1:]
                    projection_tensor = projection_tensor[:, 1:]
                    cum_sum = tf.cumsum(exp_var, axis=0)
                    sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var))
                # Condition 2: Each explained variance is greater than the desired sum, then take the minimum
                elif tf.math.greater(cum_sum[0], self.projection_var) and tf.greater_equal(tf.reduce_min(exp_var),
                                                                                           self.projection_var):
                    sel_indices = tf.argmin(exp_var)
                # Condition 3: Cumulative sum of explained variance is larger than the desired sum at mid way
                else:
                    if tf.math.greater_equal(self.projection_var, 0.5):
                        idx_tmp = tf.not_equal(tf.range(tf.shape(exp_var)[0]), tf.argmin(exp_var).numpy())
                    else:
                        idx_tmp = tf.not_equal(tf.range(tf.shape(exp_var)[0]), tf.argmax(exp_var).numpy())
                    exp_var = tf.boolean_mask(exp_var, idx_tmp)
                    projection_tensor = tf.boolean_mask(projection_tensor, idx_tmp)
                    cum_sum = tf.cumsum(exp_var, axis=0)
                    sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var))

            # Gather the selected projection tensors
            projection_tensor = tf.gather(projection_tensor, sel_indices, axis=1)
            if tf.greater(projection_tensor.ndim, 2):
                projection_tensor = tf.squeeze(projection_tensor, axis=-1)
            else:
                projection_tensor = tf.expand_dims(projection_tensor, axis=-1)

        # For when lambda is larger than 1
        else:
            # Initialize projection tensor at zero mean unit std dev distribution
            projection_tensor = tf.random.normal(shape=(channel_sz, out_ch_lim), mean=0, stddev=1, seed=channel_sz)

            # Obtain column indices of projection tensor and compute the square root of the column indices
            # while creating the extra projection tensor of the same rule
            col_indices = tf.cast(tf.range(1, out_ch_lim + 1), dtype=tf.float32)
            col_sqrt = tf.reshape(tf.sqrt(col_indices), [1, out_ch_lim])
            col_sqrt_extra = tf.reshape(tf.sqrt(1/col_indices), [1, out_ch_lim])
            projection_tensor = projection_tensor / col_sqrt
            projection_tensor_extra = projection_tensor / col_sqrt_extra

            # Shuffle the preprocessed projection tensors
            col_indices = tf.range(out_ch_lim)
            shuffled_indices = tf.random.shuffle(col_indices, seed=channel_sz)
            projection_tensor = tf.gather(projection_tensor, shuffled_indices, axis=1)
            projection_tensor_extra = tf.gather(projection_tensor_extra, shuffled_indices, axis=1)

            # Selection of shuffled extra projection tensor based on lambda
            col_var = tf.math.reduce_variance(projection_tensor_extra, axis=0)
            exp_var = col_var / tf.reduce_sum(col_var)
            cum_sum = tf.cumsum(exp_var, axis=0)
            sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var - 1))

            while tf.size(sel_indices).numpy() == 0:
                # Condition 1: First cumulative sum of explained variance already larger than the desired sum
                if tf.math.greater(cum_sum[0], self.projection_var - 1) and tf.less_equal(tf.reduce_min(exp_var),
                                                                                      self.projection_var - 11):
                    exp_var = exp_var[1:]
                    projection_tensor_extra = projection_tensor_extra[:, 1:]
                    cum_sum = tf.cumsum(exp_var, axis=0)
                    sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var - 1))
                # Condition 2: Each explained variance is greater than the desired sum, then take the minimum
                elif tf.math.greater(cum_sum[0], self.projection_var - 11) and tf.greater_equal(tf.reduce_min(exp_var),
                                                                                           self.projection_var - 1):
                    sel_indices = tf.argmin(exp_var)
                # Condition 3: Cumulative sum of explained variance is larger than the desired sum at mid way
                else:
                    if tf.math.greater_equal(self.projection_var - 1, 0.5):
                        idx_tmp = tf.not_equal(tf.range(tf.shape(exp_var)[0]), tf.argmin(exp_var).numpy())
                    else:
                        idx_tmp = tf.not_equal(tf.range(tf.shape(exp_var)[0]), tf.argmax(exp_var).numpy())
                    exp_var = tf.boolean_mask(exp_var, idx_tmp)
                    projection_tensor_extra = tf.boolean_mask(projection_tensor_extra, idx_tmp)
                    cum_sum = tf.cumsum(exp_var, axis=0)
                    sel_indices = tf.where(tf.less_equal(cum_sum, self.projection_var - 1))

            # Gather the selected extra projection tensors
            projection_tensor_extra = tf.gather(projection_tensor_extra, sel_indices, axis=1)

            # Concatenate the extra projection tensor with projection tensor of lambda=1
            projection_tensor = tf.concat([tf.expand_dims(projection_tensor, axis=-1), projection_tensor_extra], axis=1)
            if tf.greater(projection_tensor.ndim, 2):
                projection_tensor = tf.squeeze(projection_tensor, axis=-1)
            else:
                projection_tensor = tf.expand_dims(projection_tensor, axis=-1)

        if self.output_channels is None:
            # Assign to weight
            self.weight = self.add_weight(name='LinearProjectionWeights',
                                          shape=projection_tensor.shape,
                                          initializer=keras.initializers.Constant(projection_tensor),
                                          trainable=True)
            self.output_channels = self.weight.shape[1]
        else:
            self.weight = self.add_weight(name='LinearProjectionWeights',
                                          shape=(channel_sz, self.output_channels),
                                          initializer=keras.initializers.RandomNormal(),
                                          trainable=True)

        super(ProjectionLayer, self).build(input_shape)

    def call(self, inputs):
        # Perform matrix multiplication of inputs and weights
        return tf.matmul(inputs, self.weight)

    # Ensure the parameters are saved and loaded properly
    def get_config(self):
        config = super(ProjectionLayer, self).get_config()
        config.update({
            'projection_var': self.projection_var,
            'output_channels': self.output_channels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # For deserialization
        return cls(**config)


# Custom callback class (modified from ModelCheckpoint of Keras callback),
# you will need this to load back the trained model
@keras.utils.register_keras_serializable(package='Custom', name='AccuracyCallback')
class AccuracyCallback(keras.callbacks.Callback):
    def __init__(self,
                 initial_acc,
                 initial_loss,
                 start_from_epoch,
                 verbose,
                 target="loss",
                 monitor=None,
                 restore_best_weights=True,
                 save_best_only=True,
                 ):

        super(AccuracyCallback, self).__init__()
        if monitor is None:
            monitor = ['val_accuracy', 'val_loss']

        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.save_best_only = save_best_only
        self.best_weights = None
        self.best_epoch = 0
        self.best_acc = initial_acc
        self.best_loss = initial_loss
        self.target = target
        self.start_from_epoch = start_from_epoch
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best_weights = None
        self.best_epoch = 0

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                io_utils.print_msg(
                    "Restoring model weights from "
                    "the end of the best epoch: "
                    f"{self.best_epoch + 1}."
                )
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):
        current_acc, current_loss = self.get_monitor_value(logs)

        if current_acc is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # If best weights were never set,
            # then the current weights are the best.
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        if self.target == "loss":
            improve = self._loss_improvement(current_loss, self.best_loss)
            improve_text = f"\nEpoch {epoch + 1}: {self.monitor[1]} improved from {self.best_loss:.5f} to {current_loss:.5f}."
            no_improve_text = f"\nEpoch {epoch + 1}: {self.monitor[1]} did not improved from {self.best_loss}."
        else:
            improve = self._acc_improvement(current_acc, self.best_acc)
            improve_text = f"\nEpoch {epoch + 1}: {self.monitor[0]} improved from {self.best_acc:.5f} to {current_acc:.5f}."
            no_improve_text = f"\nEpoch {epoch + 1}: {self.monitor[0]} did not improved from {self.best_acc}."

        if improve:
            if self.verbose == 1:
                io_utils.print_msg(improve_text)

            self.best_acc = current_acc
            self.best_loss = current_loss
            self.best_epoch = epoch

            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        else:
            if self.verbose == 1:
                io_utils.print_msg(no_improve_text)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_acc_value = logs.get(self.monitor[0])
        monitor_loss_value = logs.get(self.monitor[1])
        return monitor_acc_value, monitor_loss_value

    def _acc_improvement(self, monitor_value, reference_value):
        return keras.ops.greater(monitor_value, reference_value)

    def _loss_improvement(self, monitor_value, reference_value):
        return keras.ops.less(monitor_value, reference_value)

    def get_config(self):
        return {}