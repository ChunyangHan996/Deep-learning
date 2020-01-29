
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential, Model, model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split

FTRAIN = 'training.csv'
FTEST = 'test.csv'
FIdLookup = 'IdLookupTable.csv'

#--------------------------CLASSES-------------------------------------

class DataModifier(object):
    def fit(self, X_, y_):
        return (NotImplementedError)


class FlipPic(DataModifier):
    def __init__(self, flip_indices=None):
        if flip_indices is None:
            flip_indices = [
                (0, 2), (1, 3),
                (4, 8), (5, 9), (6, 10), (7, 11),
                (12, 16), (13, 17), (14, 18), (15, 19),
                (22, 24), (23, 25)
            ]

        self.flip_indices = flip_indices

    def fit(self, X_batch, y_batch):

        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size // 2, replace=False)

        X_batch[indices] = X_batch[indices, :, ::-1, :]
        y_batch[indices, ::2] = y_batch[indices, ::2] * -1

        # flip left eye to right eye, left mouth to right mouth and so on ..
        for a, b in self.flip_indices:
            y_batch[indices, a], y_batch[indices, b] = (
                y_batch[indices, b], y_batch[indices, a]
            )
        return X_batch, y_batch

#--------------------------LOAD METHODS-------------------------------------

def load(test=False, cols=None):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return:
    X: 2-d numpy array (Nsample, Ncol*Nrow)
    y: 2-d numpy array (Nsample, Nlandmarks*2)
       In total there are 15 landmarks.
       As x and y coordinates are recorded, u.shape = (Nsample,30)

    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)
    ## row with at least one NA columns are removed!
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if not test:  # labels only exists for the training data
        ## standardization of the response
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # y values are between [-1,1]
        X, y = shuffle(X, y, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    re = load(test, cols)

    X = re[0].reshape(-1, 96, 96, 1)
    y = re[1]

    return X, y

#--------------------------MODELS-------------------------------------

def model_piyush(dropoutFlag = False):
    """
    Model taken from this link:
    https://github.com/piyush2896/Facial-Keypoints-Detection/blob/master/FacialPointRecognition-Kaggle.ipynb
    :param dropoutFlag: decides dropout or not
    :return: model
    """

    model = Sequential()

    model.add(BatchNormalization(input_shape=(96, 96, 1)))
    model.add(Conv2D(24, 5, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(96, 96, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Conv2D(36, 5))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Conv2D(48, 5))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    if dropoutFlag:
        model.add(Dropout(0.2))

    model.add(Conv2D(64, 3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if dropoutFlag:
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    if dropoutFlag:
        model.add(Dropout(0.2))

    model.add(Dense(90))
    model.add(Activation("relu"))

    model.add(Dense(30))
    #Optimizer outside this!
    #Compile outside this function!!

    return model

def model_fairyonice(dropoutFlag = False):
    """
    From: https://fairyonice.github.io/achieving-top-23-in-kaggles-facial-keypoints-detection-with-keras-tensorflow.html
    :param dropoutFlag: decides dropout
    :return: model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1)))
    model.add(Activation('relu'))  ## 96 - 3 + 2
    model.add(MaxPool2D(pool_size=(2, 2)))  ## 96 - (3-1)*2
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))  ##
    model.add(MaxPool2D(pool_size=(2, 2)))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(500))
    model.add(Activation('relu'))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Dense(500))
    model.add(Activation('relu'))
    if dropoutFlag:
        model.add(Dropout(0.1))

    model.add(Dense(30))
    # sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(loss="mean_squared_error", optimizer=sgd)
    return model

def model_simple(X):
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(30))
    return model

#--------------------------FIT METHODS-------------------------------------

def newGenFit(model, modifier, X_train, y_train, X_val, y_val,
              batch_size=32, epochs=100,
              print_every=10,patience=np.inf):

    #Compile model
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)  # Applies Nesterov Momentum if True
    model.compile(loss='mean_squared_error', optimizer=sgd)

    generator = ImageDataGenerator()

    history = {"loss": [], "val_loss": []}
    for e in range(epochs):
        if e % print_every == 0:
            print('Epoch {:4}:'.format(e)),
        ## -------- ##
        ## training
        ## -------- ##
        batches = 0
        loss_epoch = []
        for X_batch, y_batch in generator.flow(X_train, y_train, batch_size=batch_size):
            X_batch, y_batch = modifier.fit(X_batch, y_batch)
            hist = model.fit(X_batch, y_batch, verbose=False, epochs=1)
            loss_epoch.extend(hist.history["loss"])
            batches += 1
            if batches >= len(X_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        loss = np.mean(loss_epoch)
        history["loss"].append(loss)
        ## --------- ##
        ## validation
        ## --------- ##
        y_pred = model.predict(X_val)
        val_loss = np.mean((y_pred - y_val) ** 2)
        history["val_loss"].append(val_loss)
        if e % print_every == 0:
            print("loss - {:6.5f}, val_loss - {:6.5f}".format(loss, val_loss))
        min_val_loss = np.min(history["val_loss"])
        ## Early stopping
        # if patience is not np.Inf:
        #     if np.all(min_val_loss < np.array(history["val_loss"])[-patience:]):
        #         break
    return history

def normalFit(model, X, y, epochs, loadWeights = None):
    """
    Fits using the normal images
    :param model:
    :param X:
    :param y:
    :param epochs:
    :param loadWeights:
    :return:
    """
    if loadWeights != None:
        model.load_weights(loadWeights) #'weights.h5'

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)  # Applies Nesterov Momentum if True
    model.compile(loss='mean_squared_error', optimizer=sgd)
    hist = model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=2)
    return hist

#--------------------------PLOTS-------------------------------------

def plotHist(hist):
    """
    plots the loss and validation loss in a graph
    :param hist:
    :return:
    """
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def plot_loss(hist, name, plt, RMSE_TF=False):
    '''
    RMSE_TF: if True, then RMSE is plotted with original scale
    '''
    loss = hist['loss']
    val_loss = hist['val_loss']
    if RMSE_TF:
        loss = np.sqrt(np.array(loss)) * 48
        val_loss = np.sqrt(np.array(val_loss)) * 48

    plt.plot(loss, "--", linewidth=3, label="train:" + name)
    plt.plot(val_loss, linewidth=3, label="val:" + name)

def plotMirrorData(X_train, y_train):
    """
    Use to show how the mirror data looks like
    :param X_train:
    :param y_train:
    :return:
    """
    generator = ImageDataGenerator()
    modifier = FlipPic()

    fig = plt.figure(figsize=(7,7))

    count = 1
    for batch in generator.flow(X_train[:2], y_train[:2]):
        X_batch, y_batch = modifier.fit(*batch)

        ax = fig.add_subplot(3, 3, count, xticks=[], yticks=[])
        plot_sample(X_batch[0], y_batch[0], ax)
        count += 1
        if count == 10:
            break
    plt.show()

def plot_sample(X, y, axs):
    '''
    kaggle picture is 96 by 96
    y is rescaled to range between -1 and 1
    '''

    axs.imshow(X.reshape(96, 96), cmap="gray")
    axs.scatter(48 * y[0::2] + 48, 48 * y[1::2] + 48)

#--------------------------LOAD/SAVE-------------------------------------

def save_model(model, name):
    '''
    save model architecture and model weights
    '''
    json_string = model.to_json()
    open(name + '_architecture.json', 'w').write(json_string)
    model.save_weights(name + '_weights.h5')


def load_model(name):
    model = model_from_json(open(name + '_architecture.json').read())
    model.load_weights(name + '_weights.h5')
    return model


#--------------------------MAIN-------------------------------------

def main():

    # X, y = load()
    X, y = load2d()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)

    generator = ImageDataGenerator()
    modifier = FlipPic()


    #checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)
    # model = model_piyush(dropoutFlag=True)
    # model = model_fairyonice(dropoutFlag=True) #requires load2d()
    # model = model_simple(X) #requires load()

    # hist = dataAugFit(model,X,y,epochs=100)
    # hist = normalFit(model,X,y, epochs=10)

    # plotMirrorData(X_train,y_train) # If you want to see the mirrored data visually

    model = load_model("Fair1000strongdrop")

    hist = newGenFit(model,modifier,X_train, y_train, X_val, y_val,
                     batch_size=32, epochs=1000,print_every=100)

    save_model(model, "Fair2000drop")

    plot_loss(hist, "Fairyonice Model", plt)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # plotHist(hist)


    # model.save('myModelT1.h5')


if __name__ == '__main__':
    main()
