from numpy import loadtxt
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    x = loadtxt('formattedResults/x.txt', delimiter=',')
    y = loadtxt('formattedResults/y.txt', delimiter=',')
    model = load_model('BuiltModels/v2 (500perLayer, 100%)')

    model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])

    _, accuracy = model.evaluate(x, y)
    print('Accuracy on all data: %.2f' % (accuracy * 100))

    test_x, test_y = x[400:], y[400:]

    _, accuracy = model.evaluate(test_x, test_y)
    print('Accuracy on testing data: %.2f' % (accuracy * 100))