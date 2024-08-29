from flask import Flask, render_template, request
import numpy as np
import pickle
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MultiColumnLabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

data=pd.read_csv('./content/garments_worker_productivity.csv')
data.head()
numeric_data = data.select_dtypes(include=[float, int])
corrMatrix = numeric_data.corr()
fig, ax= plt.subplots(figsize=(15,15))
sns.heatmap(corrMatrix, annot=True, linewidths=0.5, ax=ax)
plt.show()

data.describe()

data.shape

data.info()

data.isnull().sum()

data.drop(['wip'],axis=1,inplace=True)

data['date']=pd.to_datetime(data['date'])
data.date

data['month']=data['date'].dt.month
data.drop(['date'],axis=1,inplace=True)
data.month

data['department'].value_counts()

data['department']=data['department'].apply(lambda x: 'finishing' if x.replace(" ","") == 'finishing' else 'sweing')
data['department'].value_counts()

Mcle=MultiColumnLabelEncoder.MultiColumnLabelEncoder()
data=Mcle.fit_transform(data)

x=data.drop(['actual_productivity'],axis=1)
y=data['actual_productivity']
X=x.to_numpy()
X

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=0)

model_lr=LinearRegression()
model_lr.fit(x_train, y_train)
pred_test=model_lr.predict(x_test)
print("test_MSE:",mean_squared_error(y_test,pred_test))
print("test_MAE:",mean_absolute_error(y_test,pred_test))
print("R2_Score:{}".format(r2_score(y_test,pred_test)))

model_rf=RandomForestRegressor(n_estimators=200, max_depth=5)
model_rf.fit(x_train, y_train)
pred=model_rf.predict(x_test)
print("test_MSE:",mean_squared_error(y_test,pred))
print("test_MAE:",mean_absolute_error(y_test,pred))
print("R2_Score:{}".format(r2_score(y_test,pred)))

model_xgb = xgb.XGBRegressor(n_estimators = 200, max_depth=5,learning_rate = 0.1)
model_xgb.fit(x_train, y_train)
pred3=model_xgb.predict(x_test)
print("test_MSE:",mean_squared_error(y_test,pred3))
print("test_MAE:",mean_absolute_error(y_test,pred3))
print("R2_Score:{}".format(r2_score(y_test,pred3)))

with open('model_lr.pkl', 'wb') as f:
    pickle.dump(model_lr, f)
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
with open('model_xgb.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)

app = Flask(__name__)
model = pickle.load(open('model_lr.pkl','rb'))
model = pickle.load(open('model_rf.pkl','rb'))
model = pickle.load(open('model_xgb.pkl','rb'))

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/about")
def home():
    return render_template('about.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/submit")
def home2():
    return render_template('submit.html')

@app.route("/pred", methods=['POST'])
def predict():
    # Collecting form data
    quarter = request.form['quarter']
    department = request.form['department']
    day = request.form['day']
    team = request.form['team']
    targeted_productivity = request.form['targeted_productivity']
    smv = request.form['smv']
    over_time = request.form['over_time']
    incentive = request.form['incentive']
    idle_time = request.form['idle_time']
    idle_men = request.form['idle_men']
    no_of_style_change = request.form['no_of_style_change']
    no_of_workers = request.form['no_of_workers']
    month = request.form['month']
    
    # Preparing input for model
    total = [[int(quarter), int(department), int(day), int(team),
              float(targeted_productivity), float(smv), int(over_time), int(incentive),
              float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)]]
    
    # Prediction
    prediction = model.predict(total)[0]  # Assuming this returns a scalar value
    print(f"Prediction: {prediction}")
    
    if prediction <= 0.3:
        text = 'The employee is Averagely Productive.'
    elif 0.3 < prediction <= 0.8:
        text = 'The employee is Medium Productive.'
    else:
        text = 'The employee is Highly Productive.'

    graphs = []

    # Bar chart
    plt.figure()
    categories = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
    values = [targeted_productivity, smv, over_time, idle_time]
    plt.bar(categories, [float(v) for v in values], color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Employee Productivity Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Scatter plot
    plt.figure()
    plt.scatter([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)])
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Scatter Plot of Employee Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Line plot
    plt.figure()
    plt.plot([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)], marker='o')
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Line Plot of Employee Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Pie chart
    plt.figure()
    sizes = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
    labels = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart of Employee Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Histogram
    plt.figure()
    data = np.random.randn(100)
    plt.hist(data, bins=20, color='purple')
    plt.xlabel('Random Data')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Data')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Boxplot
    plt.figure()
    data = np.random.rand(100, 5)
    plt.boxplot(data)
    plt.title('Boxplot of Random Data')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    return render_template('submit.html', prediction_text=text, graphs=graphs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
