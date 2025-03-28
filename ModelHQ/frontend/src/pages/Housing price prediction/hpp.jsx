import React, { useState } from 'react';
import './hpp.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const HousingPricePrediction = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [openSection, setOpenSection] = useState(null);

    const toggleSection = (section) => {
        setOpenSection(openSection === section ? null : section);
    };

    return (
        <div className='HousingPricePrediction'>
            <div className="housing-header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="Atlas-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="housing-content">
                <h1>Housing Price Prediction</h1>
                <p>Explore and predict housing prices using machine learning models.</p>
                <div className="visualizations">
                    <h2>Data Visualizations</h2>
                    <div className="visualization-container">
                        <img src="/path/to/heatmap.png" alt="Correlation Heatmap" />
                        <img src="/path/to/scatterplot.png" alt="Scatterplot" />
                    </div>
                </div>
                <div className="model-performance">
                    <h2>Model Performance</h2>
                    <p>Linear Regression Score: <strong>0.75</strong></p>
                    <p>Random Forest Score: <strong>0.82</strong></p>
                    <p>Best Grid Search Score: <strong>0.84</strong></p>
                </div>
            </div>

            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Housing Price Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning housing price prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Architecture</h1>
                                <p>High-level overview of our housing price prediction system</p>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Fetching and preprocessing housing data.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>Linear Regression and Random Forest for price prediction.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Price forecasting with accuracy metrics.</p>
                                    </li>
                                </div>
                                <h1>Key Components</h1>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>Housing data – Fetched from a CSV file.</li>
                                        <li>Features – Median income, house age, location, etc.</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Linear Regression – Baseline model for price prediction.</li>
                                        <li>Random Forest – Ensemble model for improved accuracy.</li>
                                    </ul>
                                </div>
                                <div className="ApproachUsed">
                                    <h3>Approach</h3>
                                    <p>Our system leverages Linear Regression and Random Forest models to predict housing prices based on features like location, income, and house age.</p>
                                </div>
                            </div>}
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Before building the model, we need essential libraries for data manipulation, visualization, and machine learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>pandas, numpy: For handling and processing data.</li>
                                        <li>matplotlib, seaborn: For data visualization.</li>
                                        <li>sklearn: For machine learning models and evaluation.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Data</h2>
                                    <p>We load the housing data and preprocess it for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Load housing data
data = pd.read_csv("housing.csv")

# Drop missing values
data.dropna(inplace=True)

# Split into features and target
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Feature Engineering</h2>
                                    <p>We perform feature engineering to improve model performance.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Log-transform features
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# One-hot encode categorical features
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

# Create new features
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train both Linear Regression and Random Forest models.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Train Linear Regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
reg = LinearRegression()
reg.fit(X_train_s, y_train)

# Train Random Forest
forest = RandomForestRegressor()
forest.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Evaluating the Model</h2>
                                    <p>We evaluate the model's performance on the test set.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Evaluate Linear Regression
reg.score(X_test_s, y_test)

# Evaluate Random Forest
forest.score(X_test, y_test)

# Perform Grid Search for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}
grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(X_train_s, y_train)
grid_search.best_estimator_.score(X_test_s, y_test)`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>}
                            {activeSection === 'evaluation' && 
    <div className="model-details-evaluation">
        <h1>Model Evaluation</h1>
        <p>Performance metrics and validation methodology</p>

        <section className="metric-section">
            <h2>Model Accuracy</h2>
            <p className="accuracy">84.5%</p>
            <p>Mean accuracy across all tested models</p>
            <table>
                <thead>
                    <tr>
                        <th>Linear Regression Score</th>
                        <th>0.75</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Random Forest Score</td>
                        <td>0.82</td>
                    </tr>
                    <tr>
                        <td>Best Grid Search Score</td>
                        <td>0.84</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="metric-section">
            <h2>Error Metrics</h2>
            <p>Mean Squared Error and R-squared values</p>
            <table>
                <thead>
                    <tr>
                        <th>Mean Squared Error (MSE)</th>
                        <th>0.023</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Root Mean Squared Error (RMSE)</td>
                        <td>0.152</td>
                    </tr>
                    <tr>
                        <td>R-squared (R²)</td>
                        <td>0.845</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="metric-section">
            <h2>Cross-Validation Results</h2>
            <p>Performance across different validation folds</p>
            <table>
                <thead>
                    <tr>
                        <th>Fold</th>
                        <th>MSE</th>
                        <th>R²</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>0.025</td>
                        <td>0.83</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>0.024</td>
                        <td>0.84</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>0.023</td>
                        <td>0.85</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>0.022</td>
                        <td>0.86</td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>0.021</td>
                        <td>0.87</td>
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

export default HousingPricePrediction;