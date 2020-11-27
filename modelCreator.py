from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import loadtxt

if __name__ == "__main__":
    x = loadtxt('formattedResults/x.txt', delimiter=',')
    y = loadtxt('formattedResults/y.txt', delimiter=',')

    train_x, train_y = x[:400], y[:400]
    test_x, test_y = x[400:], y[400:]

    model = Sequential()
    model.add(Dense(1000, input_dim=197, activation='sigmoid'))
    model.add(Dense(1000, activation='sigmoid'))
    model.add(Dense(196, activation='sigmoid'))

    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=100, batch_size=10,use_multiprocessing=True,verbose=1)

    _, accuracy = model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy * 100))

    model.save('BuiltModels/v4',overwrite=True)

    