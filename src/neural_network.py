from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.callbacks import EarlyStopping


def build_model(architecture):
    model = Sequential()
    model.add(Dense(units=architecture[1], input_dim=architecture[0], activation="relu"))
    for i in range(1, len(architecture) - 1):
        model.add(Dense(units=architecture[i], activation="relu"))

    model.add(Dense(units=architecture[-1], activation="sigmoid"))
    return model


def train_model(model, train_input, train_label,
                loss=MSE,
                optimizer="RMSprop",
                epochs=8,
                batch_size=1024,
                validation_split=0.2,
                filename=None):
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse']
                  )

    history = model.fit(train_input,
                        train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=True,
                        verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0001,
                                                 patience=2,
                                                 verbose=0,
                                                 mode='auto')
                                   ]
                        )

    if filename:
        model.save(filename)

    return history
