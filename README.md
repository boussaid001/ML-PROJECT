# 🏠 Tunisia House Price Prediction

A comprehensive machine learning application for predicting real estate prices across Tunisia. This project provides an interactive web interface for exploring the Tunisian housing market, analyzing regional trends, and making accurate price predictions using multiple ML models.

## 🌟 Features

### 1. Interactive Web Interface
- Modern, responsive design with Tunisian theme
- User authentication system
- Intuitive navigation
- Real-time data visualization

### 2. Market Analysis Tools
- Interactive map visualization of property prices
- Regional price comparisons
- Property type distribution analysis
- Price trend analysis
- Feature correlation analysis

### 3. Prediction Capabilities
- Multiple ML models:
  - Linear Regression
  - Random Forest
  - XGBoost
- Comprehensive property feature inputs
- Similar property recommendations
- Price comparison with market averages

### 4. Data Visualization
- Interactive charts and graphs
- Regional price heatmaps
- Property type distribution plots
- Feature correlation analysis
- Price distribution analysis

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: MySQL (XAMPP)
- **Machine Learning**: 
  - scikit-learn
  - XGBoost
  - pandas
  - numpy
- **Visualization**:
  - Plotly
  - Matplotlib
  - Seaborn
- **Authentication**: JWT, bcrypt

## 📋 Prerequisites

- Python 3.8+
- XAMPP (for MySQL)
- pip (Python package manager)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tunisia-house-prediction.git
cd tunisia-house-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up the database:
- Start XAMPP and ensure MySQL is running
- Create a new database named "ML"
- No password is required for the root user

## 💻 Usage

1. Generate the dataset:
```bash
python src/generate_tunisia_data.py
```

2. Train the models:
```bash
python src/train_models.py
```

3. Run the application:
```bash
streamlit run src/app.py
```

4. Access the application at `http://localhost:8504`

## 📁 Project Structure

```
tunisia-house-prediction/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── utils.py            # Utility functions
│   ├── database.py         # Database configuration
│   ├── auth.py             # Authentication functions
│   ├── generate_tunisia_data.py  # Dataset generation
│   └── train_models.py     # Model training script
├── models/                 # Trained model files
├── data/                   # Dataset directory
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔍 Features in Detail

### Data Generation
- Synthetic dataset based on Tunisian housing market characteristics
- Covers all major regions in Tunisia
- Includes various property types and features
- Realistic price ranges based on market conditions

### Model Training
- Multiple regression models for price prediction
- Feature engineering and preprocessing
- Model performance evaluation
- Cross-validation and hyperparameter tuning

### Web Interface
- User authentication and registration
- Interactive data exploration
- Real-time price predictions
- Regional market analysis
- Property type comparisons

## 📊 Model Performance

The models are evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Streamlit documentation
- scikit-learn documentation
- XGBoost documentation
- MySQL documentation

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/tunisia-house-prediction](https://github.com/yourusername/tunisia-house-prediction)
