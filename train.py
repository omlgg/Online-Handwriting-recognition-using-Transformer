import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from model import Transformer
from utils import VectorizeChar, DisplayOutputs, CustomSchedule, path_to_features , wer, cer
import pandas as pd

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")
train_df['file_path'] = train_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/'+x.split('/')[-2]+'/'+x.split('/')[-1].replace('.inkml','.bin'))
val_df['file_path'] = val_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/val/'+x.split('/')[-1].replace('.inkml','.bin'))
train_df['file_path'] = train_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/'+x.split('/')[-2]+'/'+x.split('/')[-1].replace('.inkml','.bin'))

MATHWRITING_ROOT_DIR='data/mathwriting-2024'
TRAIN_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'train')
VAL_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'valid')
TEST_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'test')
SYMBOL_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'symbols')


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def encode(path,txt):
  """Encode a text into a sequence of vectors."""
  txt = txt.numpy().decode('utf8')
  y = tf.convert_to_tensor(vectorizer(txt),dtype=tf.int64)
  x = path_to_features(path)
  return x,y
def tf_encode(path,txt):
  """ util py function to be used with tensors."""
  x,y = tf.py_function(encode, [path,txt], [tf.float32,tf.int64])
  return x,y
def create_tf_dataset(data, batch_size=4):
  """Create a tf.data.Dataset from the given data."""
  dataset = tf.data.Dataset.from_tensor_slices((np.array(data["file_path"].values),np.array(data["transcript"].values)))
  dataset = dataset.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes=([None,20], [None]))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(lambda x, y: {"source": x, "target": y})
  #dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/device:GPU:0"))
  #dataset = dataset.cache()
  return dataset


# Prepare items
set_seeds()
train_ds = create_tf_dataset(train_df, batch_size=128)
val_ds = create_tf_dataset(val_df, batch_size=8)
vectorizer = VectorizeChar(100)
max_target_len = 100



batch = next(iter(val_ds))
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2
)

# create the model and compile it
model = Transformer(
    num_hid=100,
    num_head=8,
    num_feed_forward=256,
    target_maxlen=100,
    num_layers_enc=10,
    num_layers_dec=1,
    num_classes=len(vectorizer.get_vocabulary()),
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1,
)

learning_rate = CustomSchedule(
    init_lr=0.00001,
    lr_after_warmup=0.001,
    final_lr=0.00001,
    warmup_epochs=15,
    decay_epochs=85,
    steps_per_epoch=len(train_ds),
)
optimizer = keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(train_ds, validation_data=val_ds, callbacks=[display_cb], epochs=100)

pd.DataFrame(history.history).to_csv("history.csv")
model.save_weights("model_weights.weights.h5")
