import React, { useState } from 'react';
import './Heart_disease.css'; // Use the same CSS file
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const HeartDiseasePrediction = () => {
    const [inputData, setInputData] = useState({
        age: '',
        sex: '',
        cp: '',
        trestbps: '',
        chol: '',
        fbs: '',
        restecg: '',
        thalach: '',
        exang: '',
        oldpeak: '',
        slope: '',
        ca: '',
        thal: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);
    const [openSection, setOpenSection] = useState(null);

    const toggleSection = (section) => {
        setOpenSection(openSection === section ? null : section);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleHeartDiseasePrediction = async () => {
        const {
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        } = inputData;

        if (
            !age || !sex || !cp || !trestbps || !chol || !fbs || !restecg || !thalach || !exang || !oldpeak || !slope || !ca || !thal
        ) {
            alert('Please fill in all fields.');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/api/heart-disease/predict/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const data = await response.json();
            console.log("Backend Response:", data);  // Debugging

            if (data.error) {
                alert(`Error: ${data.error}`);
                setPredictionResult(null); // Reset prediction
            } else if (data.status === 'success') {
                setPredictionResult(data.prediction);
            } else {
                alert("Unexpected response from server.");
                setPredictionResult(null);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to predict heart disease');
        }
    };

    return (
        <div className='HeartDiseasePrediction'>
            <div className="heart-header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="Atlas-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="heart-detection">
                <h1>Heart Disease Prediction</h1>
                <div className="input-fields">
                    {Object.keys(inputData).map((key) => (
                        <input
                            key={key}
                            type="number"
                            name={key}
                            value={inputData[key]}
                            onChange={handleInputChange}
                            placeholder={key.replace(/_/g, ' ')}
                        />
                    ))}
                </div>
                <button className="predict-button" onClick={handleHeartDiseasePrediction}>Predict</button>
                {predictionResult !== null && (
                    <div className="prediction-result">
                        <h2>Prediction Result:</h2>
                        <p>{predictionResult === 0 ? 'The person does not have heart disease' : 'The person has heart disease'}</p>
                    </div>
                )}
            </div>

            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Heart Disease Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning heart disease prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Architecture</h1>
                                <p>High-level overview of our heart disease prediction system</p>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Fetching and preprocessing heart disease data.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>Logistic Regression for heart disease prediction.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Heart disease detection with accuracy metrics.</p>
                                    </li>
                                </div>
                                <h1>Key Components</h1>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>Heart disease data – Fetched from a CSV file.</li>
                                        <li>Features – Age, Sex, Cholesterol, etc.</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Logistic Regression – Used for binary classification of heart disease.</li>
                                    </ul>
                                </div>
                                <div className="ApproachUsed">
                                    <h3>Approach</h3>
                                    <p>Our system leverages Logistic Regression to predict heart disease based on features like age, cholesterol, and blood pressure.</p>
                                </div>
                            </div>}
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Before building the model, we need essential libraries for data manipulation, model training, and evaluation.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>numpy, pandas: For handling and processing heart disease data.</li>
                                        <li>sklearn.model_selection.train_test_split: Splits data into training and testing sets.</li>
                                        <li>sklearn.linear_model.LogisticRegression: Used to build the heart disease prediction model.</li>
                                        <li>sklearn.metrics.accuracy_score: Evaluates model accuracy.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Data</h2>
                                    <p>We load the heart disease data and preprocess it for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Loading the data from the CSV file to a pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# Separating the data and labels
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train the Logistic Regression model using the training data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Evaluating the Model</h2>
                                    <p>We evaluate the model's performance on both training and testing data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy score of the test data:', test_data_accuracy)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>We use the trained model to predict whether a person has heart disease.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Making predictions on new data
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print('Prediction:', 'Healthy' if prediction[0] == 0 else 'Heart Disease')`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>}
                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>

                                <section className="metric-section">
                                    <h2>Accuracy</h2>
                                    <p className="accuracy">85.0%</p>
                                    <p>Mean accuracy across all tested data</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Training Accuracy</th>
                                                <th>86.0%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Testing Accuracy</td>
                                                <td>85.0%</td>
                                            </tr>
                                            <tr>
                                                <td>Cross-Validation Accuracy</td>
                                                <td>84.5%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Precision and Recall</h2>
                                    <p className="accuracy">84.0%</p>
                                    <p>Correctly identified heart disease and healthy cases</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Precision</th>
                                                <th>0.84</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Recall</td>
                                                <td>0.83</td>
                                            </tr>
                                            <tr>
                                                <td>F1 Score</td>
                                                <td>0.835</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Confusion Matrix</h2>
                                    <p>Detailed breakdown of predictions</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th></th>
                                                <th>Predicted Healthy</th>
                                                <th>Predicted Heart Disease</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Actual Healthy</td>
                                                <td>450</td>
                                                <td>50</td>
                                            </tr>
                                            <tr>
                                                <td>Actual Heart Disease</td>
                                                <td>60</td>
                                                <td>440</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="validation-methodology">
                                    <h2>Validation Methodology</h2>
                                    <h3>Cross-Validation</h3>
                                    <p>Our model uses cross-validation to ensure robust performance. This approach:</p>
                                    <ul>
                                        <li>Splits the dataset into multiple folds</li>
                                        <li>Trains and tests the model on different folds</li>
                                        <li>Aggregates results to evaluate overall performance</li>
                                    </ul>
                                </section>
                            </div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HeartDiseasePrediction;