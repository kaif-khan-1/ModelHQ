import React, { useState } from 'react';
import axios from 'axios';
import './EmployeeAttritionPrediction.css';

const EmployeeAttritionPrediction = () => {
    const [inputData, setInputData] = useState({
        Age: '',
        DailyRate: '',
        DistanceFromHome: '',
        Education: '',
        EnvironmentSatisfaction: '',
        JobInvolvement: '',
        JobLevel: '',
        JobSatisfaction: '',
        MonthlyIncome: '',
        NumCompaniesWorked: '',
        PercentSalaryHike: '',
        PerformanceRating: '',
        RelationshipSatisfaction: '',
        StockOptionLevel: '',
        TotalWorkingYears: '',
        TrainingTimesLastYear: '',
        WorkLifeBalance: '',
        YearsAtCompany: '',
        YearsInCurrentRole: '',
        YearsSinceLastPromotion: '',
        YearsWithCurrManager: '',
        Department_Sales: '',
    Department_ResearchDevelopment: '',
    EducationField_LifeSciences: '',
    EducationField_Marketing: '',
    EducationField_TechnicalDegree: ''
    });
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleAttritionPrediction = async () => {
        try {
            // Convert input values to the correct data types
            const formattedData = Object.fromEntries(
                Object.entries(inputData).map(([key, value]) => [key, parseFloat(value)])
            );

            console.log('Formatted Input Data:', formattedData); // Log the formatted input data
            const response = await axios.post('http://localhost:8000/predict/employee_attrition', formattedData);
            console.log('API Response:', response.data); // Log the API response
            setPredictionResult(response.data);
        } catch (error) {
            console.error('Error making prediction:', error);
            alert('Failed to get prediction. Please check the input or try again later.');
        }
    };

    return (
        <div className="EmployeeAttritionPrediction">
            <div className="header">
                <div className="logo">Employee Attrition Prediction</div>
            </div>
            <div className="input-container">
                {Object.keys(inputData).map((key) => (
                    <div className="input-group" key={key}>
                        <label htmlFor={key}>{key}</label>
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
            <button className="predict-button" onClick={handleAttritionPrediction}>
                Predict
            </button>
            {predictionResult && (
                <div className="prediction-result">
                    <p><strong>Prediction:</strong> {predictionResult.prediction === 1 ? 'Attrition' : 'No Attrition'}</p>
                    <p>
                        <strong>Probability:</strong> 
                        {predictionResult.probability !== undefined 
                            ? predictionResult.probability.toFixed(2) 
                            : 'N/A'}
                    </p>
                </div>
            )}
        </div>
    );
};

export default EmployeeAttritionPrediction;