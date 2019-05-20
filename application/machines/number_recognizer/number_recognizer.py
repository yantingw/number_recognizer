import keras
model = load_model('my_model.h5')
print('test after load: ', model.predict(X_test[0:2]))
