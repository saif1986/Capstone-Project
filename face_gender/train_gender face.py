

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from model.datasets import Center_control
from cnn.cnn import mini_XCEPTION
from model.datamangment import ImageLoder
from model.datasets import divide_imdb

# parameters
batch_size = 32
num_epochs = 1
validation_split = .25
do_random_crop = False
patience = 100
num_classes = 2
file_name ='imdb'
input_shape = (64, 64, 1)
if input_shape[2] == 1:
    grayscale = True
images_path = 'datasets/imdb_crop/imdb_crop/'

trained_models_path = 'trained_models/gender_models/gender_mini_XCEPTION'

# data loader
data_loader = Center_control(file_name)
Database = data_loader.obtainment()
train_keys, val_keys = divide_imdb(Database, validation_split)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))
image_generator = ImageLoder(Database, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)


model_checkpoint = ModelCheckpoint(filepath='gender_Weights/best_gender_weights.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, early_stop, reduce_lr]

# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))


