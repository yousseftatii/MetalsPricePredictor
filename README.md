# 🏆 Metals Price Predictor

![Project Banner](https://via.placeholder.com/1280x640.png/1c110c/F5DEB3?text=Metals+Price+Predictor+📈+Predict+Gold+%26+Silver+Prices+with+AI)

**An Advanced Machine Learning Platform for Precious Metals Market Forecasting**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://metal-price-predictor-yousseftatii.streamlit.app)
![License](https://img.shields.io/badge/License-MIT-gold.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blueviolet)
![ML Framework](https://img.shields.io/badge/Scikit--Learn-1.3.2-red)

## 🌟 Project Overview

**Metals Price Predictor** is a sophisticated machine learning application that forecasts gold and silver prices using advanced regression techniques. Designed for financial analysts, investors, and commodity traders, this tool combines:

- Real-time market data analysis
- Interactive scenario simulation
- Advanced Random Forest algorithms
- Comprehensive economic indicators monitoring

![App Screenshot](https://via.placeholder.com/800x400.png/1c110c/F5DEB3?text=Interactive+Web+Interface+📊)

## 🚀 Key Features

### Predictive Analytics Engine
- Dual-model architecture (Gold & Silver)
- R² scores up to 0.9857
- Multi-factor input system:
  - Foreign exchange reserves
  - Natural gas prices
  - Consumer Price Index (CPI)
  - Industrial exports

### Interactive Visualization Suite
- Real-time prediction simulations
- Dynamic parameter adjustment
- 3D market trend projections
- Historical performance analysis

### Technical Highlights
```python
# Core Prediction Logic
def predict_metal_price(target):
    model = joblib.load(f"{target}_model.pkl")
    scaler = joblib.load(f"{target}_scaler.pkl")
    scaled_input = scaler.transform(user_inputs)
    return model.predict(scaled_input)[0]
📊 Model Architecture & Performance Metrics

Metal	Algorithm	MAE	RMSE	R²
Gold	Random Forest	36.77	49.25	0.9857
Silver	Random Forest	164.79	264.98	0.9070
🛠 Installation Guide
Prerequisites
Python 3.9+
Git
Streamlit account (for deployment)
Setup Instructions
bash
Copy
Edit
# Clone repository
git clone https://github.com/yourusername/metals-price-predictor.git

# Navigate to project directory
cd metals-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
File Structure
bash
Copy
Edit
metals-price-predictor/
├── app.py                 # Main application logic
├── clean_data.csv         # Processed dataset
├── models/
│   ├── or_model.pkl       # Gold price predictor
│   └── argent_model.pkl   # Silver price predictor
├── requirements.txt       # Dependency list
└── README.md              # This documentation
🌍 Deployment
Cloud Deployment via Streamlit
Ensure all model files and data are in the root directory
Create requirements.txt with specified dependencies
Connect GitHub repo to Streamlit Cloud
Configure deployment settings:
Python version: 3.9
Startup command: streamlit run app.py
Live Demo: Metals Price Predictor Dashboard

📈 Data Pipeline Architecture
mermaid
Copy
Edit
graph TD
    A[Raw Economic Data] --> B{Data Cleaning}
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Hyperparameter Tuning]
    E --> F[Prediction Interface]
    F --> G[Visual Analytics]
📚 Technical Documentation
Key Dependencies
Package	Version	Purpose
Streamlit	1.41.1	Web interface framework
Scikit-learn	1.6.1	Machine learning models
Plotly	5.24.1	Interactive visualizations
Joblib	1.4.2	Model serialization
Feature Engineering
Logarithmic transformations for non-linear relationships
StandardScaler normalization (μ=0, σ=1)
Temporal feature encoding
Outlier detection using IQR method
📱 User Guide
Select Target Metal
Choose between Gold (Prix Or) or Silver (PrixArgent)
Adjust Market Parameters
Use interactive sliders to simulate market conditions
Generate Predictions
Click "Générer la Prédiction" for instant forecast
Analyze Results
Review predicted price
Explore prediction confidence metrics
Compare with historical trends
Usage Demo
📜 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

🤝 Contribution
Contributions are welcome! Please follow these steps:

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📧 Contact
Project Maintainers:

Youssef TATI: youssef.tati@example.com
Karim Maktouf: karim.maktouf@example.com
