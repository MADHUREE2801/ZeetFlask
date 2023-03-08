import csv
import io

from io import BytesIO, StringIO
import os
from flask import Flask, make_response,render_template,request,url_for,Response,redirect,session
import pickle
from matplotlib import transforms
import numpy as np
import pandas as pd
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from distutils.log import debug
from fileinput import filename

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from array import array

import csv
import codecs
from flask import (jsonify, request)




app=Flask(__name__,static_folder="C:/Users/Dell/Documents/Major Project frontend/login_sys/views/app/static")
model=pickle.load(open('model.pkl','rb'))

app.secret_key = 'super secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'login_crud'

mysql = MySQL(app)


@app.route('/')
def home():
  return render_template('home.hbs')
  
@app.route('/login',methods=['GET','POST'])
def login():
      msg = ''
      if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE users.email = %s AND pass = %s', (email, password, ))
        account = cursor.fetchall()
        if account:
            session['loggedin'] = True
            print(msg)
            msg = 'Logged in successfully !'
            return render_template('login.hbs',msg=msg)
        else:
            msg = 'Incorrect email / password !'
            return render_template('login.hbs',msg=msg)
      return render_template('login.hbs',msg=msg)
@app.route('/welcome')
def welcome():
  return render_template('welcome.hbs')
@app.route('/single_csv')
def single_csv():
  return render_template('single_csv.hbs')

 

@app.route('/signup',methods=['GET','POST'])
def signup():
      msg = ''
      if request.method == 'POST' and 'fullname' in request.form and 'email' in request.form and 'password' in request.form  and 'confirm_password' in request.form:
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email =% s', (email, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
            return render_template('signup.hbs',msg=msg)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z]+', fullname):
            msg = 'Fullname must contain only characters !'
        elif not fullname or not password or not email:
            msg = 'Please fill out the form !'
        elif password !=confirm_password:
          msg="Passwords don't match!"
        else:
            cursor.execute('INSERT INTO users VALUES (NULL, % s, % s, % s)', (fullname, email, password, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
      elif request.method=="POST":
        msg = 'Please fill out the form !'
      return render_template('signup.hbs', msg = msg)

  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('ID', None)
    session.pop('email', None)
    return render_template('home.hbs')





    


@app.route('/predict',methods=["GET","POST"])
def predict():
 int_features=list()
 if request.method == "POST":
  for x in request.form.values():
    if(x!="Predict Transaction"):
        int_features.append(x)
         

  final_features = [np.asarray(int_features,dtype=str)]
  print(final_features)
  prediction = model.predict(final_features)

  if prediction==0:
    return  render_template('welcome.hbs',pred='It is a legit transaction')
  else:
   return render_template('welcome.hbs',pred='It is a fraud transaction')

@app.route('/predict_fraud')
def predict_fraud():
  if 'loggedin' in session:
   return render_template('predict_fraud.hbs')
  return redirect(url_for('login'))

UPLOAD_FOLDER = os.path.join('static')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/transform', methods=["POST"])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('data_file')

 
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))
 
        uploaded_data_file_path =os.path.join(app.config['UPLOAD_FOLDER'],
                     data_filename)

        return render_template('uploaded.hbs')

    return render_template('notuploaded.hbs')

@app.route('/predict_csv',methods=['GET','POST'])
def parseCSV():
  credit_card_data=pd.read_csv('C:/Users/Dell/Documents/Major Project frontend/login_sys/views/single_csv.csv')
  Pred=model.predict(credit_card_data).round()
  if Pred==0:
    pred='It is a legit transaction'
  if Pred==1:
    pred='It is a fraud transaction'
  return render_template('uploaded.hbs',pred=pred)

@app.route('/performance')
def performance():
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 import sklearn
 from sklearn import metrics
 from sklearn.model_selection import train_test_split
 #from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import accuracy_score
 from sklearn.metrics import precision_score,recall_score,f1_score

 import tensorflow as tf
 from tensorflow import keras
 from keras import Sequential
 from keras.layers import Flatten,Dense,Dropout,BatchNormalization
 from keras.layers import Conv1D,MaxPool1D
 from keras.optimizers import Adam
 from mlxtend.plotting import plot_confusion_matrix
 from sklearn.metrics import confusion_matrix  



 import pickle                                            #converts data in python file into serialized form

 credit_card_data=pd.read_csv('C:/Users/Dell/Documents/Major Project frontend/login_sys/views/creditcard.csv')

 #Dropping missing values
 credit_card_data=credit_card_data.dropna(how='any')

 #Distributing data into legit and fraud
 legit= credit_card_data[credit_card_data.Class==0]
 fraud=credit_card_data[credit_card_data.Class==1] 

 #Under-sampling
 legit_sample=legit.sample(n=492)
 #Conactenating
 new_dataset=pd.concat([legit_sample,fraud],axis=0)

 #Splitting the data into features and target
 X=new_dataset.drop(columns='Class',axis=1)
 Y=new_dataset['Class']

 #Split the data into training data and testing data
 X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

 #Random Forest
 #rf=RandomForestClassifier()
 #rf.fit(X_train.values,Y_train.values)
 #Y_train_pred3=rf.predict(X_train.values)
 #Y_test_pred3=rf.predict(X_test.values)

 #we need to reshape the data from 2D to 3D before passing it as input to CNN
 X_train=X_train.to_numpy()
 X_test=X_test.to_numpy()
 X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
 X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

 #Build CNN

 #By increasing no of epochs and adding pooling layer we are achieving good training accuracy and less validation loss(overcoming overfitting problem )

 epochs=40
 model=Sequential()

 #First Layer
 model.add(Conv1D(32,2,activation='relu',input_shape=X_train[0].shape))
 model.add(BatchNormalization())
 model.add(MaxPool1D(2))
 model.add(Dropout(0.2))     #20% of neurons will be dropped after 1st layer

 #Second Layer
 model.add(Conv1D(64,2,activation='relu'))
 model.add(BatchNormalization())
 model.add(MaxPool1D(2))

 model.add(Dropout(0.5))     #50% of neurons will be dropped after 2nd layer

 #Flattening Layer   ----Converting multidimensional data into a vector
 model.add(Flatten())

 #Dense Layer
 model.add(Dense(64,activation='relu'))
 model.add(Dropout(0.5))

 model.add(Dense(1,activation='sigmoid'))

 model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

 #Train the model
 history=model.fit(X_train,Y_train,epochs=epochs,validation_data=(X_test,Y_test),verbose=1)

 Y_train_pred=model.predict(X_train).round()
 CNN_train_accuracy=accuracy_score(Y_train,Y_train_pred)
 CNN_train_precision=precision_score(Y_train,Y_train_pred)
 CNN_train_recall=recall_score(Y_train,Y_train_pred)
 CNN_train_f1=f1_score(Y_train,Y_train_pred)
 import itertools
 import matplotlib.pyplot as plt
 import numpy as np

 #from mlxtend.plotting import plot_confusion_matrix
 from sklearn.metrics import confusion_matrix
 #mat=confusion_matrix(Y_train_pred,Y_train)
 #plot_confusion_matrix(conf_mat=mat)
 def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

 cnf_matrix1=metrics.confusion_matrix(Y_train_pred,Y_train)
 np.set_printoptions(precision=2)

#print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
 class_names = [0,1]
 plt.figure()
 plot_confusion_matrix(cnf_matrix1
                      , classes=class_names
                      , title='Confusion matrix for Training Data')
#plot_confusion_matrix(cnf_matrix2 , classes=class_names, title='Confusion matrix')

 plt.show()
 plt.savefig('new_plot.png')


 return render_template('performance.hbs',acc=round(CNN_train_accuracy*100,2),prec=round(CNN_train_precision*100,2),
                        recall=round(CNN_train_recall*100,2),
                        f1=round(CNN_train_f1*100,2))





#def transform_view():
 #   f = request.files['data_file']
  #  if not f:
   #     return "No file"

    #stream = StringIO(f.stream.read().decode("ISO-8859-1"), newline=None)
    #csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    #print(csv_input)
    #for row in csv_input:
     #   print(row)

    #stream.seek(0)
    #result = transforms.Transform(stream.read())

    #df = pd.read_csv(StringIO(result))
    

    # load the model from disk
    #loaded_model = pickle.load(open('modeel.pkl', 'rb'))
    #df['prediction'] = loaded_model.predict(df)

    

    #response = make_response(df.to_csv())
    #response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    #return response




app.run(debug=True)