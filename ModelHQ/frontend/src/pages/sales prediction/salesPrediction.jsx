import React, { useState } from 'react';
import './salesPrediction.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';

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
                    <p>A detailed guide to our Walmart sales prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Overview</h1>
                                <p>High-level overview of our Walmart sales prediction system</p>
                            </div>}
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Details about the implementation of the Walmart sales prediction model</p>
                            </div>}
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

export default SalesPrediction;