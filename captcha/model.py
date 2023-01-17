import tensorflow as tf
import os
# import matplotlib.pyplot as plt
import numpy as np

@tf.function
def split_image(img, split_size=(64, 64)):
    img_u32 = tf.cast(img, tf.uint32)
    img_24bit = img_u32[...,0] * 2**16 + img_u32[...,1] * 2**8 + img_u32[...,2]
    img_24bit_flat = tf.reshape(img_24bit, [-1])
    y, _, count = tf.unique_with_counts(img_24bit_flat)
    # if tf.size(count) < 6:
    #     return
    _, yixs = tf.math.top_k(count, 6)
    colors = tf.gather(y, yixs[1:])

    shape = tf.shape(img)[:2]
    boxes = tf.constant([], dtype=tf.int32)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    for i in range(5):
        iscolor = tf.where(img_24bit == colors[i])
        iscolor = tf.cast(iscolor, tf.int32)
        topleft = tf.reduce_min(iscolor, axis=0)
        bottomright = tf.reduce_max(iscolor, axis=0)
        topleft = tf.math.maximum(topleft - 5, 0)
        bottomright = tf.math.minimum(bottomright + 5, shape - 1)
        # print(topleft, bottomright)
        # print(tf.concat([topleft, bottomright], axis=-1))
        boxes = tf.concat([boxes, topleft, bottomright], axis=-1)
        # target = bottomright - topleft + 1
        # plt.imshow(tf.image.crop_to_bounding_box(img, topleft[0], topleft[1], target[0], target[1]))
        # plt.show()

    boxes = tf.reshape(boxes, (5, 4))
    order = tf.argsort(boxes[:, 1])
    boxes = tf.gather(boxes, order)
    boxes /= tf.concat([shape, shape], axis=-1)
    boxes = tf.cast(boxes, tf.float32)

    splits = tf.image.crop_and_resize([img], boxes, tf.repeat(0, 5), split_size)
    splits /= 255

    return splits

    # _, ax = plt.subplots(1, 5, squeeze=False)
    # ax = ax[0]
    # for i in range(5):
    #     ax[i].imshow(splits[i])
    # plt.show()

# split_image(tf.io.decode_png(tf.io.read_file("data/train/93185.png")))
# exit()
    
if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 128
    input_size = (64, 64)
    channels = 3

    list_ds = tf.data.Dataset.list_files("data/train/*", shuffle=False)
    list_ds = list_ds.shuffle(5000, reshuffle_each_iteration=False)

    def process_path(file_path):
        img = tf.io.decode_png(tf.io.read_file(file_path))
        splits = split_image(img, split_size=input_size)
        # img = tf.image.rgb_to_grayscale(img)
        # img = tf.image.resize(img, input_size)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        if channels == 3:
            splits = tf.image.rgb_to_yuv(splits)
        elif channels == 1:
            splits = tf.image.rgb_to_grayscale(splits)

        png_name = tf.strings.split(file_path, os.path.sep)[-1]
        label = tf.strings.split(png_name, '.')[0]
        # label = tf.strings.split(label, '-')[0]
        digits = tf.strings.bytes_split(label)[:5]
        digits = tf.strings.to_number(digits, out_type=tf.int32)
        return splits, digits

    captcha_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).unbatch()

    train_ds = captcha_ds.skip(500)
    val_ds = captcha_ds.take(500)

    def configure_for_performance(ds):
      ds = ds.cache()
      ds = ds.shuffle(buffer_size=5000)
      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=AUTOTUNE)
      return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(68, 136, 3)),
        # tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        # tf.keras.layers.Conv2D(32, 1, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.build((batch_size, *input_size, channels))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_ds, validation_data=val_ds,
              epochs=100, batch_size=batch_size,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=10)])

    def predict(file_path):
        splits, _ = process_path(file_path)
        preds = model.predict(splits, batch_size=5, verbose=0)
        preds = np.argmax(preds, axis=1)
        return np.array2string(preds, separator='')[1:-1]

    for captcha in ["93185", "11430", "12741", "71730"]:
        print(predict(f"data/train/{captcha}.png"))

    model.save("model")
