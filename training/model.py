import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCH=20
TRAINING_SIZE=0.8
VALIDATION_SIZE=0.1
TEST_SIZE=0.1

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names
plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[0]])
        plt.axis("off")
        
        
        
def get_dataset_partition_tf(ds, train_ratio=0.8,valid_ratio=0.1,test_ratio=0.1,shuffle_size=10000,shuffle=True):
    ds_size=len(dataset)

    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=11)
    
    train_Dsize=np.round(ds_size*train_ratio)
    valid_Dsize=np.round(ds_size*valid_ratio)
    test_Dsize= ds_size-(train_Dsize+valid_Dsize)
    
    train_ds=dataset.take(train_Dsize)
    test_ds=dataset.skip(train_Dsize)
    valid_ds=test_ds.take(valid_Dsize)
    test_ds=test_ds.skip(valid_Dsize)
    
    return train_ds,valid_ds,test_ds

train_ds,valid_ds,test_ds=get_dataset_partition_tf(dataset)
print(len(train_ds),len(valid_ds),len(test_ds))


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds=valid_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)       


resize_and_rescaling = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(factor=(-0.25, 0.25)),  # Rotate images by up to ±25% of a full circle (±90 degrees)
    layers.RandomContrast(factor=0.02),            # Adjust contrast by a factor of up to ±20%
    # layers.RandomBrightness(factor=0.02)           # Adjust brightness by a factor of up to ±20%
])

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
model = models.Sequential([
    resize_and_rescaling, 
    data_augmentation,
    layers.Conv2D(128, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64,kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[ 'accuracy']
)

history = model.fit(
    train_ds, 
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=valid_ds
)
