import React, { useState } from 'react';
import './CarPricePrediction.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';

const CarPricePrediction = () => {
    const [inputData, setInputData] = useState({
        year: '',
        mileage: '',
        age: '',
        brand: '',
        model: '',
        title_status: '',
        color: '',
        state: '',
        country: '',
        condition: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleCarPricePrediction = async () => {
        try {
            const response = await fetch('http://localhost:8000/predict/car_price', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                setPredictionResult(`Predicted Car Price: $${result.prediction.toFixed(2)}`);
            } else {
                setPredictionResult(`Error: ${result.message}`);
            }
        } catch (error) {
            setPredictionResult('An error occurred while making the prediction.');
        }
    };

    const labels = {
        year: "Year",
        mileage: "Mileage",
        age: "Age",
        brand: "Brand",
        model: "Model",
        title_status: "Title Status",
        color: "Color",
        state: "State",
        country: "Country",
        condition: "Condition"
    };

    return (
        <div className="CarPricePrediction">
            <div className="header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="CarPricePrediction-hero">
                <h1>Car Price <span>Prediction <br /> Model</span></h1>
                <p>Our advanced AI model evaluates car details to predict its price with high accuracy.</p>
            </div>
            <div className="car-detection">
                <div className="notice">
                    <h3>Enter car details</h3>
                    <p>Enter correct data to get accurate results</p>
                </div>
                <div className="input-container">
                    {Object.keys(inputData).map((key) => (
                        <div className="input-group" key={key}>
                            <label htmlFor={key}>{labels[key]}</label>
                            <input
                                type="text"
                                id={key}
                                name={key}
                                placeholder={`Enter ${labels[key]}`}
                                value={inputData[key]}
                                onChange={handleInputChange}
                            />
                        </div>
                    ))}
                </div>
                <button className="predict-button" onClick={handleCarPricePrediction}>
                    Predict
                </button>
                {predictionResult && (
                    <div className="prediction-result">
                        {predictionResult}
                    </div>
                )}
            </div>
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Car Price Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning car price prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Overview</h1>
                                <p>High-level overview of our car price prediction system</p>
                            </div>}
                            {activeSection === 'implementation' && 
                                <div className="model-details-implementation">
                                    <h1>Car Price Prediction Implementation</h1>
                                    <p>Detailed technical breakdown of our machine learning pipeline</p>
                                    
                                    <div className="implementation-phase">
                                        <h2>1. Data Loading & Cleaning</h2>
                                        <div className="code-block">
                                            <SyntaxHighlighter language="python" style={dracula}>
                            {`# Load and clean the dataset
                            df = pd.read_csv('/content/extracted_files/USA_cars_datasets.csv')
                            df = df.drop(['Unnamed: 0', 'vin', 'lot'], axis=1)
                            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)`}
                                            </SyntaxHighlighter>
                                            <div className="code-explanation">
                                                <p><strong>Line 1:</strong> Loads the raw car dataset from CSV</p>
                                                <p><strong>Line 2:</strong> Drops unnecessary columns</p>
                                                <p><strong>Line 3:</strong> Cleans price format (removes $ and commas)</p>
                                                <p><em>Data Quality:</em> Proper cleaning ensures accurate modeling</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="implementation-phase">
                                        <h2>2. Feature Engineering</h2>
                                        <div className="code-block">
                                            <SyntaxHighlighter language="python" style={dracula}>
                            {`# Create new features
                            current_year = pd.Timestamp.now().year
                            df['age'] = current_year - df['year']
                            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')`}
                                            </SyntaxHighlighter>
                                            <div className="code-explanation">
                                                <p><strong>Line 1-2:</strong> Calculates vehicle age from current year</p>
                                                <p><strong>Line 3:</strong> Ensures mileage is numeric</p>
                                                <p><em>ML Insight:</em> Age is often more predictive than manufacture year</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="implementation-phase">
                                        <h2>3. Data Preprocessing Pipeline</h2>
                                        <div className="code-block">
                                            <SyntaxHighlighter language="python" style={dracula}>
                            {`# Numeric feature pipeline
                            numeric_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])

                            # Categorical feature pipeline
                            categorical_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])`}
                                            </SyntaxHighlighter>
                                            <div className="code-explanation">
                                                <p><strong>Line 1-3:</strong> Handles missing numeric values and scales them</p>
                                                <p><strong>Line 6-8:</strong> Processes categorical data with one-hot encoding</p>
                                                <p><em>Best Practice:</em> Pipelines ensure consistent preprocessing</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="implementation-phase">
                                        <h2>4. Model Training</h2>
                                        <div className="code-block">
                                            <SyntaxHighlighter language="python" style={dracula}>
                            {`# Complete modeling pipeline
                            lr_model = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])

                            # Train the model
                            lr_model.fit(X_train, y_train)`}
                                            </SyntaxHighlighter>
                                            <div className="code-explanation">
                                                <p><strong>Line 1-3:</strong> Combines preprocessing and linear regression</p>
                                                <p><strong>Line 6:</strong> Trains the model on prepared data</p>
                                                <p><em>ML Concept:</em> End-to-end pipelines prevent data leakage</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            }
                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>
                            </div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CarPricePrediction;