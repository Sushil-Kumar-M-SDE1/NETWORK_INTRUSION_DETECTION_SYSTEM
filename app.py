import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, render_template, request, redirect, url_for


#read the csv file

df=pd.read_csv("C:/Users/sushil/Documents/Git_projects/project/networkintrusion_modified.csv")

#split the attributes

X=df.iloc[:,:-1]

y=df.iloc[:,-1]


#split the data in training and testing 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the Model (Random Forest Classifier)

clf.fit(X_train, y_train)


app = Flask(__name__)


#user data for login
users = {'admin': '123'}

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')




@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Check if credentials match
    if username in users and users[username] == password:
        return redirect(url_for('dashboard'))
    else:
        return "Invalid login credentials!", 401

@app.route('/dashboard')
def dashboard():
    return render_template('upload.html')


@app.route('/preview')
def preview():
    return render_template('preview.html')


@app.route('/results.html', methods=['POST'])
def results():
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    return render_template('results.html', accuracy=(accuracy*100), conf_matrix=conf_matrix, class_report=class_report)

# def send_sms_alert(message, to_phone):
#     # Your Twilio account SID and Auth Token
#     account_sid = '********'  # Replace with your Twilio Account SID
#     auth_token = '******'    # Replace with your Twilio Auth Token
#     from_phone = '+12315183707'  # Replace with your Twilio phone number
    
#     # Initialize the Twilio client
#     client = Client(account_sid, auth_token)

#     # Send SMS
#     message = client.messages.create(
#         body=message,
#         from_=from_phone,
#         to=to_phone
#     )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the user inputs from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])
        feature6 = float(request.form['feature6'])
        feature7 = float(request.form['feature7'])
        feature8 = float(request.form['feature8'])
        feature9 = float(request.form['feature9'])
        feature10 = float(request.form['feature10'])
        feature11 = float(request.form['feature11'])
        feature12 = float(request.form['feature12'])
        feature13 = float(request.form['feature13'])

        # Create a numpy array from the inputs (to match the input shape of the model)
        input_features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13]])

        

        # Make prediction using the trained model
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)

        prediction = clf.predict(input_features)
        
        
        
            # message = "An intrusion has been detected on your network. Immediate action required!"
            # to_phone = "+**********"  # Replace with the user's phone number
            # send_sms_alert(message, to_phone)                                                              //optinal feature set up your twilio account
            
        return render_template('result.html', prediction=prediction[0])
        
        # Return the result to the user

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)













