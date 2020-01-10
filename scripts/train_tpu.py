import os
import sys
from time import time
import argparse
import tensorflow as tf
from utils import parse_depth, parse_bool
"""
Custom training loop example with TPU support in TF2.0.
"""


def load_data():
  """Load fashion mnist data.
  """
  (x_train, y_train), (x_test,
                       y_test) = tf.keras.datasets.fashion_mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
      len(x_train)).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

  return train_ds, test_ds


def get_model():
  tf.keras.backend.set_floatx('float64')
  return tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
  ])


class CustomLR(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, summary):
    super(CustomLR, self).__init__()
    self._initial_learning_rate = 1e-3
    self._decay_rate = 0.96
    self._decay_steps = 100000
    self._summary = summary

  def __call__(self, step):
    learning_rate = self._initial_learning_rate * self._decay_rate**(
        step / self._decay_steps)

    with self._summary.as_default():
      tf.summary.scalar(
          'learning_rate', learning_rate, step=tf.cast(step, tf.int64))

    return learning_rate


def main(hparams):
  train_ds, test_ds = load_data()

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

  # the output dir should be somewhere on the gcp bucket
  train_summary = tf.summary.create_file_writer(hparams.output_dir)
  test_summary = tf.summary.create_file_writer(hparams.output_dir + '/test')

  # initialize everything for training
  train_ds, validate_ds = load_data()
  custom_lr = CustomLR(train_summary)
  optimizer = tf.keras.optimizers.Adam(learning_rate=custom_lr)

  @tf.function
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      predictions = model(images)
      loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

  @tf.function
  def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  def train_and_validate(hparams, model, train_ds, test_ds):
    for epoch in range(hparams.epochs):
      start = time()
      for images, labels in train_ds:
        train_step(images, labels)
      elapse = time() - start

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      print(
          f'Epoch {epoch + 1} Loss: {train_loss.result()} Accuracy: {100*train_accuracy.result()} Test Loss: {test_loss.result()} Test Accuracy: {100*test_accuracy.result()}'
      )

      with train_summary.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result() * 100, step=epoch)
        tf.summary.scalar('elapse (s)', elapse, step=epoch)

      with test_summary.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result() * 100, step=epoch)

      # Reset the metrics for the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

  if hparams.use_tpu:
    # tf.compat.v2.enable_v2_behavior()
    # tf.compat.v1.disable_eager_execution()
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=hparams.tpu_worker, zone=hparams.tpu_zone)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(
        tpu_cluster_resolver=resolver)

    with strategy.scope():
      model = get_model()
      model.summary()

      train_and_validate(hparams, model, train_ds, test_ds)
  else:
    model = get_model()
    model.summary()
    train_and_validate(hparams, model, train_ds, test_ds)


def parse_args(args):
  parser = argparse.ArgumentParser(...)

  parser.add_argument(
      '--epochs',
      default=1,
      type=int,
      help='The number of epochs to train for. Default epochs=1.')
  parser.add_argument(
      '--output_dir',
      default='runs',
      type=str,
      help='Output directory to save checkpoints and training results')
  parser.add_argument(
      '--tpu_worker',
      default='',
      type=str,
      help='Cloud TPU Instance ID or Colab $TPU_WORKER, \
      e.g. grpc://{grpc://ip.address.of.tpu:8470|os.environ[COLAB_TPU_ADDR]}'                                                                             )
  parser.add_argument(
      '--use_tpu',
      default=False,
      type=parse_bool,
      help='True if using TPU. Default False.')
  parser.add_argument(
      '--tpu_zone',
      default='us-central1-f',
      type=str,
      help='TPU zone, should be same as GCP project e.g. us-central1-f.')

  hparams = parser.parse_args(args)

  return hparams


if __name__ == '__main__':
  hparams = parse_args(sys.argv[1:])
  print(hparams)
  main(hparams)