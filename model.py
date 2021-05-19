import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def generate_model():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(
        128, 128, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))

    return classifier


def train(model, train_generator, test_generator, epochs):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True
                                   )

    reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2,
                                            patience=3,
                                            verbose=1,
                                            min_delta=0.0001)

    callbacks_list = [early_stopping, reduce_learningrate]

    model.summary()
    model.compile(loss='binary_crossentropy',
                       optimizer=Adam(lr=0.001),
                       metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.n//test_generator.batch_size,
        callbacks=callbacks_list)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['accuracy'],
             label='Training Accuracy', color='green')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    _DATA_PATH_STR = 'D:\\桌面\\Python\\parkinsons\\datasets\\drawings'

    classifier = generate_model()
    # classifier = load_model('D:\\桌面\\Python\\parkinsons\\model\\spiral_model.h5')
    # classifier = load_model('D:\\桌面\\Python\\parkinsons\\model\\wave_model.h5')

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    spiral_train_generator = train_datagen.flow_from_directory(f'{_DATA_PATH_STR}/spiral/training',
                                                               target_size=(
                                                                   128, 128),
                                                               batch_size=32,
                                                               class_mode='binary')
    spiral_test_generator = test_datagen.flow_from_directory(f'{_DATA_PATH_STR}/spiral/testing',
                                                             target_size=(
                                                                 128, 128),
                                                             batch_size=32,
                                                             class_mode='binary')

    wave_train_generator = train_datagen.flow_from_directory(f'{_DATA_PATH_STR}/wave/training',
                                                             target_size=(
                                                                 128, 128),
                                                             batch_size=32,
                                                             class_mode='binary')
    wave_test_generator = test_datagen.flow_from_directory(f'{_DATA_PATH_STR}/wave/testing',
                                                           target_size=(
                                                               128, 128),
                                                           batch_size=32,
                                                           class_mode='binary')
    
    train(classifier, spiral_train_generator, spiral_test_generator, 120)
    classifier.save_weights('./model/spiral_weights_2.h5')
    classifier.save('./model/spiral_model_2.h5')

    classifier.summary()
    classifier.evaluate(spiral_test_generator)

    with open('spiral_model_summary_2.txt', 'w') as f:
        classifier.summary(print_fn=lambda x: f.write(x + '\n'))

    
    pred = classifier.predict(spiral_test_generator)
    clas = np.round(pred)
    file_names = spiral_test_generator.filenames
    
    '''
    pred = classifier.predict(wave_test_generator)
    clas = np.round(pred)
    file_names = wave_test_generator.filenames
    '''

    df = pd.DataFrame({
        'File Name': file_names,
        'Probability': pred[:, 0],
        'Predict label': clas[:, 0]
    })
    df['Predict label'].replace({0: 'Healthy', 1: 'Parkinsons\'s'}, inplace=True)

    print(df)