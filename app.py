from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tahmin',methods=['POST','GET'])
def cicek_tahmin():
    if request.method=='POST':
        
        result=request.form
        sepal_length = result['sepal_length']
        sepal_width = result['sepal_width']
        petal_length = result['petal_length']
        petal_width = result['petal_width']
        sonuclar = np.array([sepal_length, sepal_width, petal_length, petal_width])
        sonuclar = sonuclar.reshape(1,-1)
        
        from sklearn.externals import joblib
        model = joblib.load('logmodel.pkl')
        tahmin_sonucu = model.predict(sonuclar)
        return render_template('result.html', prediction = tahmin_sonucu)

    
if __name__ == '__main__':
	app.debug = True
	app.run()