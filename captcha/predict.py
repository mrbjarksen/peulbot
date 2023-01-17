import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
from model import split_image

def predict(img):
    splits = split_image(img)
    splits_yuv = tf.image.rgb_to_yuv(splits)
    model = tf.keras.models.load_model("model")
    preds = model.predict(splits_yuv, batch_size=5, verbose=0)
    preds = np.argmax(preds, axis=1)
    pred = np.array2string(preds, separator='')[1:-1]

    _, ax = plt.subplot_mosaic('AAAAA;BCDEF')
    ax['A'].imshow(img, interpolation='nearest')
    ax['A'].set_xticks([]); ax['A'].set_yticks([])
    for i, c in enumerate('BCDEF'):
        ax[c].imshow(splits[i], interpolation='nearest')
        ax[c].set_xticks([]); ax[c].set_yticks([])

    plt.figtext(0.5, 0.05, "Prediction: " + pred, ha='center', fontsize=18)

    plt.show()

def get_captcha(from_file=None):
    if from_file is not None:
        img_raw = tf.io.read_file(from_file)
    else:
        img_raw = requests.get("https://projecteuler.net/captcha/show_captcha.php").content
    return tf.image.decode_png(img_raw)

if __name__ == "__main__":
    from_file = None
    for _ in range(100):
        predict(get_captcha(from_file))
