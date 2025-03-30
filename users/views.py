from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os
import pickle
import pandas as pd
import numpy as np

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHome.html', {})
def ada(request):
    return render(request,'users/Ada.html',{})
def xg(request):
    return render(request,'users/xg.html',{})
def gradent(request):
    return render(request,'users/gradent.html',{})
def pr(request):
    return render(request,'users/pr.html',{})
def DatasetView(request):
    from django.conf import settings
    # path = settings.MEDIA_ROOT + "//" + 'balanced_kyphosis_dataset.csv'
    path = settings.MEDIA_ROOT + "//" + 'gooddata.csv'
    
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})
from sklearn.svm import SVC
############################################################
#------------adaboodsting----------------#
def ml_training(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle  # Import for saving the model

    # Load the dataset
    # df = settings.MEDIA_ROOT + "//" + 'balanced_kyphosis_dataset.csv'
    path = settings.MEDIA_ROOT + "//" + 'gooddata.csv'
    
    df = pd.read_csv(path)
    
    
    # df = pd.read_csv(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\media\gooddata.csv')
    # df = pd.read_csv(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\media\balanced_kyphosis_dataset.csv')

    # Features and target variable
    X = df.drop('Kyphosis', axis=1)  # Features
    y = df['Kyphosis']               # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model
    model = DecisionTreeClassifier(max_depth=1)  # Weak learner
    model1 = AdaBoostClassifier(base_estimator=model, n_estimators=50, learning_rate=1.0, random_state=42)

    # Train the model
    model1.fit(X_train_scaled, y_train)   
    
    
    # Define paths for saving the model and scaler in the media folder
    model_path = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
    scaler_path = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')

    # Save the model and scaler to files
    with open(model_path, 'wb') as f:
        pickle.dump(model1, f)
        
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    
    

    # # Save the model to a file
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    # Make predictions
    y_pred = model1.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    CR = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    categories = list(report_dict.keys())[:-3]  # Ignore 'accuracy', 'macro avg', 'weighted avg'
    precision = [report_dict[cat]['precision'] for cat in categories]
    recall = [report_dict[cat]['recall'] for cat in categories]
    f1_score = [report_dict[cat]['f1-score'] for cat in categories]

    # Create Bar Plot for Classification Report
    x = np.arange(len(categories))
    width = 0.3  

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="lightblue")
    plt.bar(x, recall, width, label="Recall", color="lightgreen")
    plt.bar(x + width, f1_score, width, label="F1-Score", color="darkblue")

    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Classification Report")
    plt.xticks(x, categories)
    plt.legend()
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")
    # Save Classification Report Plot
    class_report_path = os.path.join(static_dir, "classification_report.jpg")
    plt.savefig(class_report_path)
    plt.close()

    # # Plot feature importances
    # features = X.columns
    # importances = model.feature_importances_
    # indices = np.argsort(importances)

    # plt.figure()
    # plt.title('Feature Importances')
    # plt.barh(range(X.shape[1]), importances[indices], align='center')
    # plt.yticks(range(X.shape[1]), features[indices])
    # plt.xlabel('Importance')
    # plt.show()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Ensure STATICFILES_DIRS exists
    #static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")

    # Define path to save confusion matrix
    conf_matrix_path = os.path.join(static_dir, 'confusion_matrix.jpg')

    # Save the plot
    plt.savefig(conf_matrix_path)
    plt.close()
    return render(request, 'users/ml_results.html', {'accuracy': accuracy, 'CR': CR,"conf_matrix_url": "/static/confusion_matrix.jpg","class_report_url": "/static/classification_report.png"})
    # Plot confusion matrix


def predict_kyphosis(request):
    import pandas as pd
    import pickle
    result = None
    if request.method == 'POST':
        # Load the saved model
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'rb') as f:
        #     model = pickle.load(f)

        # # Load the saved scaler
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
            
        # Save the model and scaler to files
        model_path = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
        scaler_path = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model1 = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)              

        # Extract input data from the form
        Age = float(request.POST['Age'])
        Number = int(request.POST['Number'])
        Start = int(request.POST['Start'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[Age, Number, Start]], columns=['Age', 'Number', 'Start'])

        # Standardize the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model1.predict(input_data_scaled)
        print(prediction)
        # Map the prediction to a readable format 
        if prediction[0] == 'absent':
            result = 'No Kyphosis'
        else:
            result = 'Kyphosis'
        # result = ' No Kyphosis' if prediction[0] == 1 else 'Kyphosis'
    return render(request,'users/prediction_results.html',{'result':result})

    # return render(request, 'users/predict_kyphosis.html', {'result': result})
def ml_training1(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle  # Import for saving the model

    # Load the dataset
    # df = settings.MEDIA_ROOT + "//" + 'balanced_kyphosis_dataset.csv'
    path = settings.MEDIA_ROOT + "//" + 'gooddata.csv'
    
    df = pd.read_csv(path)
    df['Kyphosis'] = df['Kyphosis'].map({'absent': 0, 'present': 1})
    
    # df = pd.read_csv(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\media\gooddata.csv')
    # df = pd.read_csv(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\media\balanced_kyphosis_dataset.csv')

    # Features and target variable
    X = df.drop('Kyphosis', axis=1)  # Features
    y = df['Kyphosis']               # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model
     # Initialize the XGBoost model
    model1 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Train the model
    model1.fit(X_train_scaled, y_train)   
    
    
    # Define paths for saving the model and scaler in the media folder
    model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
    scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')

    # Save the model and scaler to files
    with open(model_path1, 'wb') as f:
        pickle.dump(model1, f)
        
    with open(scaler_path1, 'wb') as f:
        pickle.dump(scaler, f)
    
    
    

    # # Save the model to a file
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    # Make predictions
    y_pred = model1.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    CR = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    categories = list(report_dict.keys())[:-3]  # Ignore 'accuracy', 'macro avg', 'weighted avg'
    precision = [report_dict[cat]['precision'] for cat in categories]
    recall = [report_dict[cat]['recall'] for cat in categories]
    f1_score = [report_dict[cat]['f1-score'] for cat in categories]

    # Create Bar Plot for Classification Report
    x = np.arange(len(categories))
    width = 0.3  

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="lightblue")
    plt.bar(x, recall, width, label="Recall", color="lightgreen")
    plt.bar(x + width, f1_score, width, label="F1-Score", color="darkblue")

    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Classification Report")
    plt.xticks(x, categories)
    plt.legend()
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")
    # Save Classification Report Plot
    class_report_path = os.path.join(static_dir, "classification_report.jpg")
    plt.savefig(class_report_path)
    plt.close()

    # # Plot feature importances
    # features = X.columns
    # importances = model.feature_importances_
    # indices = np.argsort(importances)

    # plt.figure()
    # plt.title('Feature Importances')
    # plt.barh(range(X.shape[1]), importances[indices], align='center')
    # plt.yticks(range(X.shape[1]), features[indices])
    # plt.xlabel('Importance')
    # plt.show()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Ensure STATICFILES_DIRS exists
    #static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")

    # Define path to save confusion matrix
    conf_matrix_path = os.path.join(static_dir, 'confusion_matrix.jpg')

    # Save the plot
    plt.savefig(conf_matrix_path)
    plt.close()
    return render(request, 'users/ml_results1.html', {'accuracy': accuracy, 'CR': CR,"conf_matrix_url": "/static/confusion_matrix.jpg","class_report_url": "/static/classification_report.png"})
    # Plot confusion matrix


def predict_kyphosis1(request):
    import pandas as pd
    import pickle
    result = None
    if request.method == 'POST':
        # Load the saved model
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'rb') as f:
        #     model = pickle.load(f)

        # # Load the saved scaler
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
            
        # Save the model and scaler to files
        model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
        scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
        
        with open(model_path1, 'rb') as f:
            model1 = pickle.load(f)

        with open(scaler_path1, 'rb') as f:
            scaler = pickle.load(f)              

        # Extract input data from the form
        Age = float(request.POST['Age'])
        Number = int(request.POST['Number'])
        Start = int(request.POST['Start'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[Age, Number, Start]], columns=['Age', 'Number', 'Start'])

        # Standardize the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model1.predict(input_data_scaled)
        print(prediction)
        # Map the prediction to a readable format 
        result = 'No Kyphosis' if prediction[0] == 0 else 'Kyphosis'
        # result = ' No Kyphosis' if prediction[0] == 1 else 'Kyphosis'
    return render(request,'users/prediction_results1.html',{'result':result})

    # return render(request, 'users/predict_kyphosis.html', {'result': result})

    # return render(request, 'users/predict_kyphosis.html', {'result': result})

#############################################################################################################################################
#----------gradient boosting---------#
def ml_training2(request):
    import os
    import pandas as pd
    import numpy as np
    from django.conf import settings
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier#<------------------->
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle

    # Load the dataset
    path = os.path.join(settings.MEDIA_ROOT, 'gooddata.csv')
    df = pd.read_csv(path)
    df['Kyphosis'] = df['Kyphosis'].map({'absent': 0, 'present': 1})

    # Features and target variable
    X = df.drop('Kyphosis', axis=1)  # Features
    y = df['Kyphosis']               # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Gradient Boosting model
    model1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model1.fit(X_train_scaled, y_train)  # Train the model

    # Define paths for saving the model and scaler
    model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
    scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')

    # Save the model and scaler
    with open(model_path1, 'wb') as f:
        pickle.dump(model1, f)
    with open(scaler_path1, 'wb') as f:
        pickle.dump(scaler, f)

    # Make predictions
    y_pred = model1.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    CR = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Extract Precision, Recall, F1-Score
    categories = list(report_dict.keys())[:-3]  # Ignore 'accuracy', 'macro avg', 'weighted avg'
    precision = [report_dict[cat]['precision'] for cat in categories]
    recall = [report_dict[cat]['recall'] for cat in categories]
    f1_score = [report_dict[cat]['f1-score'] for cat in categories]

    # Plot Classification Report Metrics
    x = np.arange(len(categories))
    width = 0.3  

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="lightblue")
    plt.bar(x, recall, width, label="Recall", color="lightgreen")
    plt.bar(x + width, f1_score, width, label="F1-Score", color="darkblue")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Classification Report")
    plt.xticks(x, categories)
    plt.legend()

    # Save Classification Report Plot
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")
    class_report_path = os.path.join(static_dir, "classification_report.jpg")
    plt.savefig(class_report_path)
    plt.close()

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save Confusion Matrix Plot
    conf_matrix_path = os.path.join(static_dir, 'confusion_matrix.jpg')
    plt.savefig(conf_matrix_path)
    plt.close()

    # Return results to the template
    return render(request, 'users/ml_results2.html', {
        'accuracy': accuracy,
        'CR': CR,
        "conf_matrix_url": "/static/confusion_matrix.jpg",
        "class_report_url": "/static/classification_report.jpg"
    })




def predict_kyphosis2(request):
    import pandas as pd
    import pickle
    result = None
    if request.method == 'POST':
        # Load the saved model
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'rb') as f:
        #     model = pickle.load(f)

        # # Load the saved scaler
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
            
        # Save the model and scaler to files
        model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
        scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
        
        with open(model_path1, 'rb') as f:
            model1 = pickle.load(f)

        with open(scaler_path1, 'rb') as f:
            scaler = pickle.load(f)              

        # Extract input data from the form
        Age = float(request.POST['Age'])
        Number = int(request.POST['Number'])
        Start = int(request.POST['Start'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[Age, Number, Start]], columns=['Age', 'Number', 'Start'])

        # Standardize the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model1.predict(input_data_scaled)
        print(prediction)
        # Map the prediction to a readable format 
        result = 'No Kyphosis' if prediction[0] == 0 else 'Kyphosis'
        # result = ' No Kyphosis' if prediction[0] == 1 else 'Kyphosis'
    return render(request,'users/prediction_results2.html',{'result':result})

    # return render(request, 'users/predict_kyphosis.html', {'result': result})

def ml_training3(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle  # Import for saving the model

    # Load the dataset
    # df = settings.MEDIA_ROOT + "//" + 'balanced_kyphosis_dataset.csv'
    path = settings.MEDIA_ROOT + "//" + 'gooddata.csv'
    
    df = pd.read_csv(path)
    df['Kyphosis'] = df['Kyphosis'].map({'absent': 0, 'present': 1})
    
    # df = pd.read_csv(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\media\gooddata.csv')
    # df = pd.read_csv(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\media\balanced_kyphosis_dataset.csv')

    # Features and target variable
    X = df.drop('Kyphosis', axis=1)  # Features
    y = df['Kyphosis']               # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the model
     # Initialize the XGBoost model
    model1 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Train the model
    model1.fit(X_train_scaled, y_train)   
    
    
    # Define paths for saving the model and scaler in the media folder
    model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
    scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')

    # Save the model and scaler to files
    with open(model_path1, 'wb') as f:
        pickle.dump(model1, f)
        
    with open(scaler_path1, 'wb') as f:
        pickle.dump(scaler, f)
    
    
    

    # # Save the model to a file
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl','wb') as f:
    # # with open(r'C:\Users\Admin\Desktop\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    # Make predictions
    y_pred = model1.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    CR = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    categories = list(report_dict.keys())[:-3]  # Ignore 'accuracy', 'macro avg', 'weighted avg'
    precision = [report_dict[cat]['precision'] for cat in categories]
    recall = [report_dict[cat]['recall'] for cat in categories]
    f1_score = [report_dict[cat]['f1-score'] for cat in categories]

    # Create Bar Plot for Classification Report
    x = np.arange(len(categories))
    width = 0.3  

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="lightblue")
    plt.bar(x, recall, width, label="Recall", color="lightgreen")
    plt.bar(x + width, f1_score, width, label="F1-Score", color="darkblue")

    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Classification Report")
    plt.xticks(x, categories)
    plt.legend()
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")
    # Save Classification Report Plot
    class_report_path = os.path.join(static_dir, "classification_report.jpg")
    plt.savefig(class_report_path)
    plt.close()

    # # Plot feature importances
    # features = X.columns
    # importances = model.feature_importances_
    # indices = np.argsort(importances)

    # plt.figure()
    # plt.title('Feature Importances')
    # plt.barh(range(X.shape[1]), importances[indices], align='center')
    # plt.yticks(range(X.shape[1]), features[indices])
    # plt.xlabel('Importance')
    # plt.show()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Ensure STATICFILES_DIRS exists
    #static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")

    # Define path to save confusion matrix
    conf_matrix_path = os.path.join(static_dir, 'confusion_matrix.jpg')

    # Save the plot
    plt.savefig(conf_matrix_path)
    plt.close()
    return render(request, 'users/ml_results1.html', {'accuracy': accuracy, 'CR': CR,"conf_matrix_url": "/static/confusion_matrix.jpg","class_report_url": "/static/classification_report.png"})
    # Plot confusion matrix
def predict_kyphosis3(request):
    import pandas as pd
    import pickle
    result = None
    if request.method == 'POST':
        # Load the saved model
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\kyphosis_model.pkl', 'rb') as f:
        #     model = pickle.load(f)

        # # Load the saved scaler
        # with open(r'C:\Users\MMC\Downloads\Kyphosis\KyphosisDiseaseClassification\scaler.pkl', 'rb') as f:
        #     scaler = pickle.load(f)
            
        # Save the model and scaler to files
        model_path1 = os.path.join(settings.MEDIA_ROOT, 'kyphosis_model.pkl')
        scaler_path1 = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
        
        with open(model_path1, 'rb') as f:
            model1 = pickle.load(f)

        with open(scaler_path1, 'rb') as f:
            scaler = pickle.load(f)              

        # Extract input data from the form
        Age = float(request.POST['Age'])
        Number = int(request.POST['Number'])
        Start = int(request.POST['Start'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[Age, Number, Start]], columns=['Age', 'Number', 'Start'])

        # Standardize the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model1.predict(input_data_scaled)
        print(prediction)
        # Map the prediction to a readable format 
        result = 'No Kyphosis' if prediction[0] == 0 else 'Kyphosis'
        # result = ' No Kyphosis' if prediction[0] == 1 else 'Kyphosis'
    return render(request,'users/pr.html',{'result':result})

def compare_models(request):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from django.conf import settings
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load dataset
    path = os.path.join(settings.MEDIA_ROOT, 'gooddata.csv')
    df = pd.read_csv(path)
    df['Kyphosis'] = df['Kyphosis'].map({'absent': 0, 'present': 1})

    # Features and target
    X = df.drop('Kyphosis', axis=1)
    y = df['Kyphosis']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
                                 use_label_encoder=False, eval_metric='logloss'),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    # Store metrics
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_scores = {}

    # Evaluate each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracies[name] = accuracy_score(y_test, y_pred)
        precisions[name] = precision_score(y_test, y_pred, average='binary')
        recalls[name] = recall_score(y_test, y_pred, average='binary')
        f1_scores[name] = f1_score(y_test, y_pred, average='binary')

    # Set static directory
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, "static")

    # Define function to plot and save graphs
    def plot_and_save(metric_dict, title, filename, color_list):
        plt.figure(figsize=(8, 6))
        model_names = list(metric_dict.keys())
        metric_values = list(metric_dict.values())
        bars = plt.bar(model_names, metric_values, color=color_list)
        plt.ylim(0, 1)
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(title.split(" ")[0])  # e.g., Accuracy, Precision, etc.
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', fontsize=12)
        graph_path = os.path.join(static_dir, filename)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

    # Plot and save all four comparison graphs
    plot_and_save(accuracies, "Accuracy Comparison of ML Models", "accuracy_comparison.jpg", ['skyblue', 'lightgreen', 'salmon'])
    plot_and_save(precisions, "Precision Comparison of ML Models", "precision_comparison.jpg", ['plum', 'lightcoral', 'gold'])
    plot_and_save(recalls, "Recall Comparison of ML Models", "recall_comparison.jpg", ['deepskyblue', 'orchid', 'limegreen'])
    plot_and_save(f1_scores, "F1 Score Comparison of ML Models", "f1score_comparison.jpg", ['tomato', 'mediumseagreen', 'cornflowerblue'])

    return render(request, 'users/comp.html', {
        'acc_graph_url': '/static/accuracy_comparison.jpg',
        'prec_graph_url': '/static/precision_comparison.jpg',
        'recall_graph_url': '/static/recall_comparison.jpg',
        'f1_graph_url': '/static/f1score_comparison.jpg'
    })


def dataset_view(request):
    path = settings.MEDIA_ROOT + "//" + 'gooddata.csv'
    df = pd.read_csv(path)  # Load dataset
    columns = df.columns.tolist()  # Extract column names
    data = df.values.tolist()  # Convert DataFrame rows to a list

    return render(request, 'users/viewdataset.html', {"columns": columns, "data": data})
