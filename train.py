from yolov3.yolo_v3_model import create_yolo_v3
from yolov3.data_loader import datagenerator, datagenerator_val
import tensorflow as tf
from yolov3.losses import getloss
from tensorflow.keras.callbacks import ReduceLROnPlateau
from yolov3.config import batch_size, epochs, anchors, num_class, step_per_epoch, step_per_val, checkpoint_path, initial_learning_rate
from tensorflow.keras.optimizers.schedules import CosineDecay


try:
    yolo_model = tf.keras.models.load_model(checkpoint_path)
    print("load a pre model yolo v3 !")
except:
    yolo_model = create_yolo_v3()
    print("create a new model yolo v3 !")

decay_steps = epochs * step_per_epoch
cosine_decay_schedule = CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=0.0
)

optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay_schedule)
yolo_model.compile(optimizer=optimizer, loss=[getloss(num_class, anchors[0]),
                                              getloss(num_class,anchors[1]),
                                              getloss(num_class,anchors[2])], run_eagerly=False)

dataset_train = tf.data.Dataset.from_generator(
    lambda: datagenerator(),
    output_signature=(
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(13, 13, 3, 5+num_class), dtype=tf.float32),  # head 1
            tf.TensorSpec(shape=(26, 26, 3, 5+num_class), dtype=tf.float32),  # head 2
            tf.TensorSpec(shape=(52, 52, 3, 5+num_class), dtype=tf.float32)  # head 3
        )))

dataset_train = dataset_train.shuffle(batch_size * 4)
dataset_train = dataset_train.repeat()
dataset_train = dataset_train.batch(batch_size)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)


dataset_val = tf.data.Dataset.from_generator(
    lambda: datagenerator_val(),
    output_signature=(
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(13, 13, 3, 5+num_class), dtype=tf.float32),  # head 1
            tf.TensorSpec(shape=(26, 26, 3, 5+num_class), dtype=tf.float32),  # head 2
            tf.TensorSpec(shape=(52, 52, 3, 5+num_class), dtype=tf.float32)  # head 3
        )))
dataset_val = dataset_val.batch(batch_size)
dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


yolo_model.fit(dataset_train, epochs=epochs, steps_per_epoch=step_per_epoch,
               validation_data=dataset_val, validation_steps=step_per_val,
               callbacks=[model_checkpoint_callback])