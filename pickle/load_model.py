import pickle

loaded_model = pickle.load(open('dib_70.pkl','rb'))
pred = loaded_model.predict([[10,20,30,40,50,10,20,10]])
print(pred)

if pred[0] ==1:
    print('Person is diabitic')
else:
    print('person is not diabitic')