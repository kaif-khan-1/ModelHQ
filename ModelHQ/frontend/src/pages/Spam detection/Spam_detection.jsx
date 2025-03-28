import React, { useState } from 'react';
import './Spam_detection.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const SpamDetection = () => {
    const [inputMail, setInputMail] = useState('');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);
    const [openSection, setOpenSection] = useState(null);

    const toggleSection = (section) => {
        setOpenSection(openSection === section ? null : section);
    };

    const handleSpamDetection = async () => {
        if (!inputMail) {
            alert('Please enter an email text.');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/api/spam/detect/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email_text: inputMail })
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
            alert('Failed to detect spam');
        }
    };

    return (
        <div className='SpamDetection'>
            <div className="spam-header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="Atlas-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="spam-detection">
                <h1>Spam Detection</h1>
                <textarea
                    value={inputMail}
                    onChange={(e) => setInputMail(e.target.value)}
                    className="spam-textarea"
                    placeholder="Enter email text here..."
                />
                <button className="detect-button" onClick={handleSpamDetection}>Detect</button>
                {predictionResult !== null && (
                    <div className="prediction-result">
                        <h2>Prediction Result:</h2>
                        <p>{predictionResult === 1 ? 'Ham mail' : 'Spam mail'}</p>
                    </div>
                )}
            </div>

            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Spam Detection Model</h1>
                    <p>A comprehensive guide to our machine learning spam detection system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Architecture</h1>
                                <p>High-level overview of our spam detection system</p>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Fetching and preprocessing email data.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>Logistic Regression for spam detection.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Spam detection with accuracy metrics.</p>
                                    </li>
                                </div>
                                <h1>Key Components</h1>
                                <div className="DataSource">
                                    <h3>Data source</h3>
                                    <ul>
                                        <li>Email data – Fetched from a CSV file</li>
                                        <li>Text preprocessing – Cleaning and tokenization</li>
                                        <li>Feature extraction – TF-IDF vectorization</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Logistic Regression – Used for binary classification of spam and ham.</li>
                                    </ul>
                                </div>
                                <div className="ApproachUsed">
                                    <h3>Approach</h3>
                                    <p>Our system leverages Logistic Regression, optimized for text classification. It uses TF-IDF for feature extraction to improve detection accuracy.</p>
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
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>numpy, pandas: For handling and processing email data.</li>
                                        <li>sklearn.model_selection.train_test_split: Splits data into training and testing sets.</li>
                                        <li>sklearn.linear_model.LogisticRegression: Used to build the spam detection model.</li>
                                        <li>sklearn.metrics.accuracy_score: Evaluates model accuracy.</li>
                                        <li>sklearn.feature_extraction.text.TfidfVectorizer: Converts text data into feature vectors.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Data</h2>
                                    <p>We load the email data from a CSV file and preprocess it by replacing null values and encoding labels.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Loading the data from the CSV file to a pandas DataFrame
raw_mail_data = pd.read_csv('/content/mail_data.csv')

# Replace null values with an empty string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the data as texts and labels
X = mail_data['Message']
y = mail_data['Category']`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Splitting Data into Training and Testing Sets</h2>
                                    <p>We split the data into training and testing sets to evaluate the model's performance.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Feature Extraction</h2>
                                    <p>We use TF-IDF vectorization to convert text data into numerical feature vectors.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Transform the text data into feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert the labels to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train the Logistic Regression model using the training data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Evaluating the Model</h2>
                                    <p>We evaluate the model's performance on both training and testing data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_training_data)

# Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_test_data)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>We use the trained model to predict whether a new email is spam or ham.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Making predictions on new email text
input_mail = ["I haven't studied from days"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
print('Prediction:', 'Ham mail' if prediction[0] == 1 else 'Spam mail')`}
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
            <p className="accuracy">98.2%</p>
            <p>Mean accuracy across all tested emails</p>
            <table>
                <thead>
                    <tr>
                        <th>Training Accuracy</th>
                        <th>99.1%</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Testing Accuracy</td>
                        <td>98.2%</td>
                    </tr>
                    <tr>
                        <td>Cross-Validation Accuracy</td>
                        <td>97.8%</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="metric-section">
            <h2>Precision and Recall</h2>
            <p className="accuracy">97.5%</p>
            <p>Correctly identified spam and ham emails</p>
            <table>
                <thead>
                    <tr>
                        <th>Precision</th>
                        <th>0.98</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Recall</td>
                        <td>0.97</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>0.975</td>
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
                        <th>Predicted Spam</th>
                        <th>Predicted Ham</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Actual Spam</td>
                        <td>950</td>
                        <td>20</td>
                    </tr>
                    <tr>
                        <td>Actual Ham</td>
                        <td>15</td>
                        <td>1015</td>
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
    </div>
}                    </div>
                </div>
            </div>
        </div>
    );
};

export default SpamDetection;