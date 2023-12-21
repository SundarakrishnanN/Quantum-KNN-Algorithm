from flask import Flask, redirect, render_template, request, session
import knn
app = Flask(__name__)
@app.route('/',methods=["GET", "POST"])
def index():
    test_data=[[]]
    if(request.method=="POST"):
        button_clicked = request.form.get('btn')
        if(button_clicked=="custom"):
            test_data[0].append(float(request.form.get('p1')))
            test_data[0].append(float(request.form.get('p2')))
            test_data[0].append(float(request.form.get('p3')))
            test_data[0].append(float(request.form.get('p4')))
            print(test_data)
            print("custom")
            prediction,classicalPred=knn.predict(test_data)
            print("classical ",classicalPred)
            print("quantum ",prediction)
            return render_template("index.html",label=None,prediction=prediction,classicalPred=classicalPred,loading="False")
            
        else:
            
            label,prediction,classicalPred=knn.predict()
            return render_template("index.html",label=label,prediction=prediction,classicalPred=classicalPred,loading="False")
            
    
        ##return render_template("index.html",loading="Predicting.....")
        
    return render_template("index.html")


