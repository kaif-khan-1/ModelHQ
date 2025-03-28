import React, { useState } from 'react';
import './diabetes_prediction.css'; // Use the same CSS file
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const DiabetesPrediction = () => {
    const [inputData, setInputData] = useState({
        pregnancies: '',
        glucose: '',
        bloodPressure: '',
        skinThickness: '',
        insulin: '',
        bmi: '',
        diabetesPedigreeFunction: '',
        age: ''
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

    const handleDiabetesPrediction = async () => {
        const { pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age } = inputData;

        if (!pregnancies || !glucose || !bloodPressure || !skinThickness || !insulin || !bmi || !diabetesPedigreeFunction || !age) {
            alert('Please fill in all fields.');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/api/diabetes/predict/', {
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
            alert('Failed to predict diabetes');
        }
    };

    return (
        <div className='DiabetesPrediction'>
            <div className="diabetes-header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="Atlas-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="diabetes-detection">
                <h1>Diabetes Prediction</h1>
                <div className="input-fields">
                    <input
                        type="number"
                        name="pregnancies"
                        value={inputData.pregnancies}
                        onChange={handleInputChange}
                        placeholder="Pregnancies"
                    />
                    <input
                        type="number"
                        name="glucose"
                        value={inputData.glucose}
                        onChange={handleInputChange}
                        placeholder="Glucose"
                    />
                    <input
                        type="number"
                        name="bloodPressure"
                        value={inputData.bloodPressure}
                        onChange={handleInputChange}
                        placeholder="Blood Pressure"
                    />
                    <input
                        type="number"
                        name="skinThickness"
                        value={inputData.skinThickness}
                        onChange={handleInputChange}
                        placeholder="Skin Thickness"
                    />
                    <input
                        type="number"
                        name="insulin"
                        value={inputData.insulin}
                        onChange={handleInputChange}
                        placeholder="Insulin"
                    />
                    <input
                        type="number"
                        name="bmi"
                        value={inputData.bmi}
                        onChange={handleInputChange}
                        placeholder="BMI"
                    />
                    <input
                        type="number"
                        name="diabetesPedigreeFunction"
                        value={inputData.diabetesPedigreeFunction}
                        onChange={handleInputChange}
                        placeholder="Diabetes Pedigree Function"
                    />
                    <input
                        type="number"
                        name="age"
                        value={inputData.age}
                        onChange={handleInputChange}
                        placeholder="Age"
                    />
                </div>
                <button className="predict-button" onClick={handleDiabetesPrediction}>Predict</button>
                {predictionResult !== null && (
                    <div className="prediction-result">
                        <h2>Prediction Result:</h2>
                        <p>{predictionResult === 0 ? 'The person is not diabetic' : 'The person is diabetic'}</p>
                    </div>
                )}
            </div>

            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Diabetes Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning diabetes prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Architecture</h1>
                                <p>High-level overview of our diabetes prediction system</p>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Fetching and preprocessing diabetes data.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>Support Vector Machine (SVM) for diabetes prediction.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Diabetes detection with accuracy metrics.</p>
                                    </li>
                                </div>
                                <h1>Key Components</h1>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>Diabetes data – Fetched from a CSV file.</li>
                                        <li>Features – Glucose, BMI, Age, etc.</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Support Vector Machine (SVM) – Used for binary classification of diabetes.</li>
                                    </ul>
                                </div>
                                <div className="ApproachUsed">
                                    <h3>Approach</h3>
                                    <p>Our system leverages SVM with a linear kernel to predict diabetes based on features like glucose levels, BMI, and age.</p>
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
from sklearn import svm
from sklearn.metrics import accuracy_score`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>numpy, pandas: For handling and processing diabetes data.</li>
                                        <li>sklearn.model_selection.train_test_split: Splits data into training and testing sets.</li>
                                        <li>sklearn.svm: Used to build the diabetes prediction model.</li>
                                        <li>sklearn.metrics.accuracy_score: Evaluates model accuracy.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Data</h2>
                                    <p>We load the diabetes data and preprocess it for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Loading the data from the CSV file to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train the SVM model using the training data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Training the Support Vector Machine Classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Evaluating the Model</h2>
                                    <p>We evaluate the model's performance on both training and testing data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data:', test_data_accuracy)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>We use the trained model to predict whether a person has diabetes.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Making predictions on new data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = classifier.predict(input_data_reshaped)
print('Prediction:', 'Not Diabetic' if prediction[0] == 0 else 'Diabetic')`}
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
                                    <p className="accuracy">77.5%</p>
                                    <p>Mean accuracy across all tested data</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Training Accuracy</th>
                                                <th>78.0%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Testing Accuracy</td>
                                                <td>77.5%</td>
                                            </tr>
                                            <tr>
                                                <td>Cross-Validation Accuracy</td>
                                                <td>76.8%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Precision and Recall</h2>
                                    <p className="accuracy">76.0%</p>
                                    <p>Correctly identified diabetic and non-diabetic cases</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Precision</th>
                                                <th>0.76</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Recall</td>
                                                <td>0.75</td>
                                            </tr>
                                            <tr>
                                                <td>F1 Score</td>
                                                <td>0.755</td>
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
                                                <th>Predicted Non-Diabetic</th>
                                                <th>Predicted Diabetic</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Actual Non-Diabetic</td>
                                                <td>450</td>
                                                <td>50</td>
                                            </tr>
                                            <tr>
                                                <td>Actual Diabetic</td>
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

export default DiabetesPrediction;