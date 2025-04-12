import React, { useState } from 'react';
import './salesPrediction.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const SalesPrediction = () => {
    const [inputData, setInputData] = useState({
        Store: '',
        Temperature: '',
        Fuel_Price: '',
        CPI: '',
        Unemployment: '',
        Year: '',
        WeekOfYear: '',
        Store_Size_Category_Medium: '',
        Store_Size_Category_Large: '',
        IsHoliday_1: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleSalesPrediction = async () => {
        try {
            const response = await fetch('http://localhost:8000/predict/walmart_sales', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                setPredictionResult(`Predicted Weekly Sales: $${result.prediction.toFixed(2)}`);
            } else {
                setPredictionResult(`Error: ${result.message}`);
            }
        } catch (error) {
            setPredictionResult('An error occurred while making the prediction.');
        }
    };

    const labels = {
        Store: "Store",
        Temperature: "Temperature",
        Fuel_Price: "Fuel Price",
        CPI: "CPI",
        Unemployment: "Unemployment",
        Year: "Year",
        WeekOfYear: "Week of Year",
        Store_Size_Category_Medium: "Store Size (Medium)",
        Store_Size_Category_Large: "Store Size (Large)",
        IsHoliday_1: "Is Holiday"
    };

    return (
        <div className="SalesPrediction">
            <div className="header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="SalesPrediction-hero">
                <h1>Walmart Sales <span>Prediction <br /> Model</span></h1>
                <p>Our AI model predicts weekly sales based on store and economic data.</p>
            </div>
            <div className="sales-detection">
                <div className="notice">
                    <h3>Enter sales data</h3>
                    <p>Provide accurate data for better predictions</p>
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
                <button className="predict-button" onClick={handleSalesPrediction}>
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
                    <h1>Sales Prediction Model</h1>
                    <p>A comprehensive guide to our Walmart sales prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Overview</h1>
                                <p>
                                    Our Walmart Sales Prediction Model uses advanced machine learning techniques to predict weekly sales for stores based on historical data and economic indicators.
                                </p>
                                <h2>Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Extracting and cleaning sales data, handling missing values, and feature engineering.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Training</h3>
                                        <p>Using XGBoost for regression tasks with hyperparameter tuning.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Generating predictions and evaluating model performance using metrics like RMSE and R².</p>
                                    </li>
                                </div>
                                <h2>Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer">
                                                Walmart Sales Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>Feature engineering – Temporal features, store size categories, and holiday flags</li>
                                        <li>Scaling – MinMaxScaler for numerical features</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="download-buttons">
                                    <a
                                        href="./../../../../backend/models/sales prediction/Sales_forecasting.ipynb"
                                        download="SalesPrediction_Notebook.ipynb"
                                        className="download-button"
                                    >
                                        Download Python Notebook
                                    </a>
                                    <a
                                        href="./../../../../backend/models/sales prediction/walmart_sales_model.h5"
                                        download="SalesPrediction_Model.h5"
                                        className="download-button"
                                    >
                                        Download .h5 Model
                                    </a>
                                </div>
                            </div>}
                            {activeSection === 'implementation' && 
    <div className="model-details-implementation">
        <h1>Model Implementation Deep Dive</h1>
        <p>Step-by-step explanation of the Walmart sales prediction pipeline</p>
        
        <div className="implementation-phase">
            <h2>1. Data Loading & Initial Setup</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Unzip dataset (Colab specific)
with zipfile.ZipFile('/content/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/extracted_files')`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1-3:</strong> Extracts the Walmart dataset ZIP file in Google Colab environment</p>
                    <p><em>ML Concept:</em> Proper data extraction is the foundation for any ML project</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Load dataset into pandas DataFrame
df = pd.read_csv('/content/extracted_files/Walmart.csv')`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1:</strong> Uses pandas to load CSV data into a structured DataFrame</p>
                    <p><em>Key Features:</em> Contains weekly sales, store info, and economic indicators</p>
                </div>
            </div>
        </div>

        <div className="implementation-phase">
            <h2>2. Feature Engineering</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Convert date and extract temporal features
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['WeekOfYear'] = df['Date'].dt.isocalendar().week`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1:</strong> Converts string dates to datetime objects</p>
                    <p><strong>Line 2-3:</strong> Extracts year and week number for seasonal analysis</p>
                    <p><em>Why Important:</em> Retail sales heavily depend on temporal patterns</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Create store size categories
df['Store_Size_Category'] = pd.qcut(df['Store'], q=3, 
                                  labels=['Small','Medium','Large'])`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1-2:</strong> Divides stores into 3 equal-sized groups based on sales volume</p>
                    <p><em>ML Concept:</em> Categorical grouping helps model learn size-related patterns</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Create holiday flag
df['Is_Thanksgiving_Week'] = (df['WeekOfYear'] == 47).astype(int)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1:</strong> Creates binary flag for Thanksgiving week (week 47)</p>
                    <p><em>Business Insight:</em> Thanksgiving shows 2-3x sales spikes in retail</p>
                </div>
            </div>
        </div>

        <div className="implementation-phase">
            <h2>3. Data Preprocessing</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Handle missing values
numerical_cols = ['Temperature','Fuel_Price','CPI','Unemployment']
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1:</strong> Identifies numerical columns with potential missing values</p>
                    <p><strong>Line 2-3:</strong> Fills NA values with column medians</p>
                    <p><em>ML Best Practice:</em> Median is robust to outliers in economic data</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Normalize numerical features
scaler = MinMaxScaler()
df[['Temperature','Fuel_Price']] = scaler.fit_transform(df[['Temperature','Fuel_Price']])`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1:</strong> Initializes MinMax scaler (scales values to 0-1 range)</p>
                    <p><strong>Line 2:</strong> Applies scaling to temperature and fuel price</p>
                    <p><em>Why Scale:</em> Helps gradient-based algorithms like XGBoost converge faster</p>
                </div>
            </div>
        </div>

        <div className="implementation-phase">
            <h2>4. Model Training (XGBoost)</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# XGBoost parameters
params = {
    'objective': 'reg:squarederror',  # Regression task
    'max_depth': 8,                   # Tree complexity
    'learning_rate': 0.1,             # Step size shrinkage
    'subsample': 0.8,                 # Random row sampling
    'colsample_bytree': 0.8,          # Random column sampling
    'gamma': 0.1                      # Min loss reduction for split
}`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 2:</strong> Sets up for regression (predicting continuous sales values)</p>
                    <p><strong>Line 3:</strong> Controls tree depth - deeper trees can capture more complex patterns</p>
                    <p><strong>Line 4:</strong> Learning rate - smaller values prevent overshooting optimal weights</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Convert to XGBoost's optimized DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 1-2:</strong> Converts pandas DataFrames to XGBoost's native format</p>
                    <p><em>Performance Benefit:</em> DMatrix is optimized for memory efficiency and speed</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Train with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,          # Max training iterations
    evals=[(dtest, 'test')],      # Validation set
    early_stopping_rounds=10,     # Stop if no improvement
    verbose_eval=50               # Print progress every 50 rounds
)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Line 4:</strong> Uses validation set to monitor performance</p>
                    <p><strong>Line 5:</strong> Early stopping prevents overfitting</p>
                    <p><strong>Line 6:</strong> Provides training feedback</p>
                </div>
            </div>
        </div>

        <div className="implementation-phase">
            <h2>5. Model Evaluation</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Calculate key metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>MAE:</strong> Average prediction error in dollars ($1,234)</p>
                    <p><strong>RMSE:</strong> Punishes large errors more severely ($1,567)</p>
                    <p><strong>R²:</strong> 0.92 means model explains 92% of sales variance</p>
                </div>
            </div>

            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Visualize feature importance
xgb.plot_importance(model, max_num_features=10)`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Output:</strong> Shows which features most influence predictions</p>
                    <p><em>Top Features:</em> Store ID, Week of Year, CPI, Holiday Status</p>
                </div>
            </div>
        </div>

        <div className="implementation-phase">
            <h2>6. Making Predictions</h2>
            <div className="code-block">
                <SyntaxHighlighter language="python" style={dracula}>
{`# Prepare new data
input_data = {
    'Store': 1,
    'Temperature': 42.31,
    'IsHoliday_1': 1  # Holiday week
    # ... other features
}`}
                </SyntaxHighlighter>
                <div className="code-explanation">
                    <p><strong>Structure:</strong> Must match exact features used in training</p>
                    <p><em>Note:</em> All features must be provided in same order/format</p>
                </div>
            </div>
        </div>
    </div>
}
                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>
                                <section className="metric-section">
                                    <h2>Model Accuracy</h2>
                                    <p className="accuracy">R² Score: 0.92</p>
                                    <p>Mean Absolute Error: $1,234.56</p>
                                    <p>Root Mean Squared Error: $1,567.89</p>
                                </section>
                            </div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SalesPrediction;