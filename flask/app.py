from flask import Flask,render_template,request
import xgboost as xgb
import numpy as np
import pandas as pd 
import pickle
import warnings
warnings.filterwarnings("ignore")

model=pickle.load(open('training\projectPGP.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/about.html')
def about_page():
    return render_template('about.html')

@app.route('/team.html')
def team_page():
    return render_template('team.html')

@app.route('/contact.html')
def contact_page():
    return render_template('contact.html')

@app.route('/services.html',methods=["GET","POST"])
def predict():
    if request.method=="GET":
        return render_template('services.html')
    else:  
        int_features=[[float(x) for x in request.form.values()]]
        print(int_features)
        
        final=np.array(int_features)
        print(final.shape)
        
        pred=model.predict_proba(final)
        print(pred[0])
        
        if(pred[0][1]<0.5):
            
            return render_template('services.html',pred_text="is Fraud")
    
        else:
            return render_template('services.html',pred_text="is not Fraud")
            

if __name__=="__main__":
    app.run(debug=True)