# Streaming-Activity-Exploration

A comprehensive data science project for exploring music streaming activity with Exploratory Data Analysis (EDA), Machine Learning (ML), and Deep Learning (DL) techniques.

## 📊 Overview

This project analyzes streaming activity data to uncover listening patterns, preferences, and trends. It includes:

- **Exploratory Data Analysis (EDA)**: Comprehensive visualization and statistical analysis of listening behavior
- **Machine Learning (ML)**: Clustering, recommendation systems, and predictive models
- **Deep Learning (DL)**: Neural networks for preference prediction and feature learning
- **AI Insights**: Pattern recognition and intelligent recommendations

## 🎵 Dataset

The project uses two main datasets:

1. **My Streaming Activity.csv**: Contains streaming history with timestamps, songs, artists, and albums
2. **Scrobble_Features.csv**: Contains Spotify audio features (danceability, energy, valence, tempo, etc.)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Streaming-Activity-Exploration.git
cd Streaming-Activity-Exploration
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook streaming_analysis.ipynb
```

2. Run all cells sequentially to perform the complete analysis

## 📈 Features

### 1. Exploratory Data Analysis (EDA)
- **Basic Statistics**: Total songs, unique artists, date range analysis
- **Top Artists & Songs**: Most played content identification
- **Temporal Analysis**: Listening patterns by hour, day, week, and month
- **Audio Features Distribution**: Analysis of danceability, energy, valence, etc.
- **Correlation Analysis**: Relationships between audio features

### 2. Machine Learning Models
- **K-Means Clustering**: Grouping songs by audio characteristics
- **Recommendation System**: Content-based recommendations using cosine similarity
- **Random Forest Regressor**: Predicting song popularity based on audio features
- **Feature Importance Analysis**: Understanding which features drive preferences

### 3. Deep Learning Models
- **Neural Network Classifier**: Predicting high-preference songs
- **Autoencoder**: Dimensionality reduction and feature learning
- **Performance Metrics**: Accuracy, AUC, and loss tracking

### 4. AI Insights
- **Listening Behavior Patterns**: Heatmaps of activity by time and day
- **Music Taste Evolution**: Tracking feature preference changes over time
- **Pattern Recognition**: Identifying listening habits and trends

## 📊 Visualizations

The notebook generates various visualizations including:
- Bar charts for top artists and songs
- Time series plots for temporal trends
- Heatmaps for correlation and activity patterns
- Distribution plots for audio features
- Cluster visualizations using PCA
- Neural network training curves
- 3D scatter plots for autoencoder features

## 🤖 Models & Algorithms

- **Clustering**: K-Means with optimal k selection via elbow method
- **Recommendation**: Cosine similarity-based content filtering
- **Regression**: Random Forest for play count prediction
- **Classification**: Deep neural network with dropout regularization
- **Feature Learning**: Autoencoder for dimensionality reduction

## 📝 Key Findings

The analysis provides insights into:
- Peak listening hours and days
- Most popular artists and songs
- Average audio feature preferences
- Song clusters based on characteristics
- Predictive models for song popularity
- Evolution of music taste over time

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning models
- **Jupyter**: Interactive analysis environment

## 📦 Requirements

See `requirements.txt` for a complete list of dependencies.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👤 Author

**yadavanujkumar**

## ⭐ Show your support

Give a ⭐️ if this project helped you!
