from flask import Flask
from flask import render_template
from flask import redirect, url_for, request
from combined_final import main


app = Flask(__name__)

@app.route('/')
def home():
   return render_template("Home.html")

@app.route('/loading')
def loading():
   return render_template("Loading.html")

@app.route('/predict')
def predict():
   a,b,c,d=main()
   b=round(b,2)
   c=round(c,2)

   return render_template("Prediction.html",a=a,b=b,c=c,d=d)

if __name__ == '__main__':
   app.run(debug = True)

