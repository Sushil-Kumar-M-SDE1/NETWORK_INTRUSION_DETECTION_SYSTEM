import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df=pd.read_csv("C:/Users/sushil/Documents/Git_projects/project/networkintrusion.csv")
features_to_drop = ['dst_host_srv_diff_host_rate','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','serror_rate','land','protocol_type','service','flag','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','num_root','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login'] 
# Drop the unwanted features
df.drop(columns=features_to_drop,inplace=True)
#selecting the X and y attributes
X=df.iloc[:,0:-1]
y=df.iloc[:,-1]
#spliting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#define the randomforest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Step 4: Train the Model (Random Forest Classifier)
clf.fit(X_train, y_train)

with open('model.pkl','wb') as f:
    pickle.dump(clf,f)
with open('model.pkl','rb') as f:
    pickle.load(f)
