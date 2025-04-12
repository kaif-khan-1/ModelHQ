import React, { useState } from 'react';
import axios from 'axios';
import './BankChernPrediction.css';

const BankChurnPrediction = () => {
    const [inputData, setInputData] = useState({
        CreditScore: '',
        Age: '',
        Tenure: '',
        Balance: '',
        NumOfProducts: '',
        HasCrCard: '',
        IsActiveMember: '',
        EstimatedSalary: '',
        Geography_Germany: '',
        Geography_Spain: '',
        Gender_Male: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleBankChurnPrediction = async () => {
        try {
            // Validate input data
            const requiredFields = Object.keys(inputData);
            for (const field of requiredFields) {
                if (inputData[field] === '' || inputData[field] === null) {
                    alert(`Please fill in the ${labels[field]} field.`);
                    return;
                }
            }
    
            // Convert input values to the correct data types
            const formattedData = {
                CreditScore: parseInt(inputData.CreditScore, 10),
                Age: parseInt(inputData.Age, 10),
                Tenure: parseInt(inputData.Tenure, 10),
                Balance: parseFloat(inputData.Balance),
                NumOfProducts: parseInt(inputData.NumOfProducts, 10),
                HasCrCard: parseInt(inputData.HasCrCard, 10),
                IsActiveMember: parseInt(inputData.IsActiveMember, 10),
                EstimatedSalary: parseFloat(inputData.EstimatedSalary),
                Geography_Germany: parseInt(inputData.Geography_Germany, 10),
                Geography_Spain: parseInt(inputData.Geography_Spain, 10),
                Gender_Male: parseInt(inputData.Gender_Male, 10),
            };
    
            console.log('Formatted Input Data:', formattedData); // Log the formatted input data
            const response = await axios.post('http://localhost:8000/predict/bank_churn', formattedData);
            console.log('API Response:', response.data); // Log the API response
            setPredictionResult(response.data);
        } catch (error) {
            console.error('Error making prediction:', error);
            alert('Failed to get prediction. Please check the input or try again later.');
        }
    };

    const labels = {
        CreditScore: 'Credit Score',
        Age: 'Age',
        Tenure: 'Tenure',
        Balance: 'Balance',
        NumOfProducts: 'Number of Products',
        HasCrCard: 'Has Credit Card (1: Yes, 0: No)',
        IsActiveMember: 'Is Active Member (1: Yes, 0: No)',
        EstimatedSalary: 'Estimated Salary',
        Geography_Germany: 'Geography (Germany: 1, Else: 0)',
        Geography_Spain: 'Geography (Spain: 1, Else: 0)',
        Gender_Male: 'Gender (Male: 1, Female: 0)'
    };

    return (
        <div className="BankChurnPrediction">
            <div className="header">
                <div className="logo">Bank Churn Prediction</div>
                <div className="book-icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                    📘
                </div>
            </div>
            <div className="input-container">
                {Object.keys(inputData).map((key) => (
                    <div className="input-group" key={key}>
                        <label htmlFor={key}>{labels[key]}</label>
                        <input
                            type="text"
                            id={key}
                            name={key}
                            value={inputData[key]}
                            onChange={handleInputChange}
                        />
                    </div>
                ))}
            </div>
            <button className="predict-button" onClick={handleBankChurnPrediction}>
                Predict
            </button>
            {predictionResult && (
                <div className="prediction-result">
                    <p><strong>Prediction:</strong> {predictionResult.prediction === 1 ? 'Churn' : 'No Churn'}</p>
                    <p>
                        <strong>Probability:</strong> 
                        {predictionResult.probability !== undefined 
                            ? predictionResult.probability.toFixed(2) 
                            : 'N/A'}
                    </p>
                </div>
            )}
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <div className="close-icon" onClick={() => setIsSidebarOpen(false)}>✖</div>
                <h3>Model Details</h3>
                <p>This model predicts whether a customer is likely to churn based on various features such as credit score, age, balance, and more.</p>
                <h4>Input Features:</h4>
                <ul>
                    {Object.values(labels).map((label, index) => (
                        <li key={index}>{label}</li>
                    ))}
                </ul>
                <h4>How it works:</h4>
                <p>The model uses a trained XGBoost classifier to predict churn probability. It preprocesses the input data, applies scaling, and outputs the prediction.</p>
            </div>
        </div>
    );
};

export default BankChurnPrediction;