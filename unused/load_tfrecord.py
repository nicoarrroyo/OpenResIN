# https://www.tensorflow.org/tutorials/load_data/tfrecord
tfrecord_path = ("C:\\Users\\User\\Documents\\paper\\Downloads\\Sentinel 2"
                 "\\S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_"
                 "20250301T152054.SAFE\\training data\\training_data.tfrecord")
import tensorflow as tf

# =============================================================================
# for example in tf.io.tf_record_iterator(tfrecord_path):
#     print(tf.train.Example.FromString(example))
# =============================================================================

data = tf.data.TFRecordDataset(tfrecord_path)
n = 10
for raw_record in data.take(n):
    print(repr(raw_record))