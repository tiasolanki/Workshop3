from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Extract feature values from query parameters
    features = request.args.getlist('feature', type=float)
    if len(features) != 4:
        return "Please provide exactly 4 features.", 400
    
    # Predict and return the result
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
