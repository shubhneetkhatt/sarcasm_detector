from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        model = joblib.load("logreg.pkl")
        
        #Unpickle vectorizer
        vectorizer = joblib.load("vectorizer.pkl")

        # Get values through input bars
        headline = request.form.get("headline")
        #weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[headline]], columns = ["headline"])
        
        #vectorize input
        #vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
        X=vectorizer.transform(X['headline'])
		
        # Get prediction
        #prediction =  X.shape
        prediction =  model.predict(X) 
		
        pred = "Not Sarcastic"
        if prediction == [1]:
            pred = "BAZINGA!!! Sarcasm Detected!!"
		
        
    else:
        pred = "Welcome to Shubhneet's Sarcasm Detector."
        
    return render_template("index.html", output = pred)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)