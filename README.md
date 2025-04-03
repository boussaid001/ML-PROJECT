# 🏠 Tunisia House Price Prediction

A state-of-the-art machine learning application for predicting real estate prices across Tunisia. This project delivers an interactive web interface for exploring the Tunisian housing market, analyzing regional trends, and generating accurate price predictions using optimized ML models.

## 🌟 Features

### 1. Interactive Web Interface
- Modern, responsive design with Tunisian theme
- Comprehensive user authentication system
- Personalized user profiles with activity tracking
- Intuitive navigation with multi-page layout
- Real-time data visualization dashboard

### 2. Market Analysis Tools
- Interactive map visualization of property prices across Tunisia
- Regional price comparisons with statistical breakdowns
- Property type distribution analysis by region
- Historical price trend analysis
- Advanced feature correlation insights
- Customizable filtering options

### 3. Prediction Capabilities
- Multiple optimized ML models:
  - Linear Regression (R² Score: 0.86)
  - Random Forest (R² Score: 0.94)
  - XGBoost (R² Score: 0.98)
- Comprehensive property feature inputs (22 features)
- Similar property recommendations
- Price comparison with market averages
- Prediction history tracking

### 4. Data Visualization
- Interactive charts and graphs with Plotly
- Regional price heatmaps
- Property type distribution plots
- Feature importance visualization
- Price distribution analysis
- Correlation heatmaps

### 5. User Profile Features
- Personalized user dashboard
- Saved property searches history
- Saved prediction history
- Favorite properties management
- User activity statistics

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
- **Data Processing**: pandas, NumPy

## 📋 Prerequisites

- Python 3.8+
- XAMPP (for MySQL)
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)

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
- No password is required for the root user (or update database.py with your credentials)

## 💻 Usage

1. Generate the dataset (10,000 properties):
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

## 🧪 Dataset

The project uses a synthetically generated dataset with realistic Tunisian housing market characteristics:

- **Size**: 10,000 property listings
- **Coverage**: All 24 regions of Tunisia
- **Property Types**: Apartments, Houses, Villas, Duplexes, Penthouses, Studios
- **Features**: 22 property attributes including:
  - Location (region, neighborhood, coordinates)
  - Property characteristics (type, age, area, bedrooms, bathrooms)
  - Amenities (elevator, garden, parking)
  - Quality indicators (construction quality, energy rating)
  - Price metrics (total price, price per square meter)

Dataset generation includes:
- Population-weighted regional distribution
- Realistic price correlations with property features
- Authentic Tunisian addresses and neighborhoods
- Strategic outlier introduction for model robustness
- Statistically sound feature relationships

## 📊 Model Performance

The latest model training results (80/20 train-test split):

| Model             | RMSE      | MAE       | R² Score |
|-------------------|-----------|-----------|----------|
| XGBoost           | 120,053   | 58,567    | 0.979    |
| Random Forest     | 211,350   | 98,924    | 0.936    |
| Linear Regression | 308,353   | 186,562   | 0.864    |

### Feature Importance

**XGBoost Top Features**:
- Property Type (Duplex): 80.5%
- Area (sqm): 3.6%
- Property Type (Penthouse): 3.5%
- Property Type (Villa): 2.1%

**Random Forest Top Features**:
- Area (sqm): 28.0%
- Property Type (Duplex): 27.0%
- Bedrooms: 9.0%
- Bathrooms: 9.0%
- Floor: 4.8%

## 📁 Project Structure

```
tunisia-house-prediction/
├── src/
│   ├── app.py                  # Main Streamlit application
│   ├── pages/                  # Application pages
│   │   ├── ExploreMarketPage.py    # Market exploration page
│   │   ├── PredictionPage.py       # Price prediction page
│   │   ├── ProfilePage.py          # User profile page
│   │   ├── AboutPage.py            # About page
│   │   └── config.py               # Configuration settings
│   ├── components/             # UI components
│   ├── utils.py                # Utility functions
│   ├── database.py             # Database configuration and functions
│   ├── auth.py                 # Authentication functions
│   ├── generate_tunisia_data.py    # Dataset generation
│   ├── train_models.py         # Model training script
│   └── config.py               # Configuration settings
├── models/                     # Trained model files
├── data/                       # Dataset directory
├── plots/                      # Generated visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🔍 Features in Detail

### Data Generation
- Synthetic dataset based on Tunisian housing market characteristics
- Population-weighted region distribution
- 6 different property types with realistic feature distributions
- Property features correlated with location and type
- Price influenced by multiple factors (area, location, amenities)
- Outlier introduction for model robustness

### Model Training
- Multiple regression models with optimized hyperparameters
- Feature engineering and preprocessing
- One-hot encoding for categorical variables
- Feature importance analysis
- Model performance evaluation
- Cross-validation and hyperparameter tuning

### Web Interface
- User authentication and registration
- Interactive data exploration with filtering
- Real-time price predictions with model selection
- Regional market analysis with visualizations
- User profile with saved searches and predictions
- Saved property management

## 👤 User Profile System

The application includes a comprehensive user profile system:

1. **User Dashboard**
   - Profile information
   - Activity statistics
   - Quick access to saved items

2. **Search History**
   - Saved property searches
   - Search parameters
   - Timestamp and results count

3. **Prediction History**
   - Saved property predictions
   - Input parameters
   - Multiple model results
   - Comparison with actual prices

4. **Saved Properties**
   - Favorite property listings
   - Quick comparison tools
   - Export functionality

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

- Streamlit community for component examples
- scikit-learn and XGBoost documentation
- Tunisian open data resources
- MySQL documentation
- Plotly and Seaborn visualization examples

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/tunisia-house-prediction](https://github.com/yourusername/tunisia-house-prediction)
