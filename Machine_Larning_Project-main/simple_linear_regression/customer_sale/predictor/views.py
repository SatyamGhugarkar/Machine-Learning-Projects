from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

# Load dataset and train model globally (once)
data = pd.read_csv('predictor/customer_sales_large.csv')
X = data[['visits']]
y = data['sales']
model = LinearRegression()
model.fit(X, y)

def index(request):
    prediction = None
    if request.method == 'POST':
        visits = float(request.POST.get('visits'))
        prediction = model.predict([[visits]])[0]
        #model.predict([visits].values.reshape(1,1)) 

    # Plotting (optional)
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel("Customer Visits")
    plt.ylabel("Sales")
    plt.title("Customer Visits vs Sales")

    # Save plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render(request, 'predictor/index.html', {'prediction': prediction, 'plot_data': plot_data})
