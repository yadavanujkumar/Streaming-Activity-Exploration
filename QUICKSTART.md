# Quick Start Guide

## Overview
This guide helps you get started with the Streaming Activity Exploration project.

## Running the Analysis

### Option 1: Jupyter Notebook (Recommended)
For interactive analysis with detailed visualizations:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open streaming_analysis.ipynb and run all cells
```

### Option 2: Python Script
For quick automated analysis:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python streaming_analysis.py
```

## What the Analysis Includes

### 1. Exploratory Data Analysis (EDA)
- **Basic Statistics**: Total songs, unique artists, temporal coverage
- **Top Content**: Most played artists and songs
- **Temporal Patterns**: When you listen most (hour, day, month)
- **Audio Features**: Distribution of song characteristics

### 2. Machine Learning (ML)
- **K-Means Clustering**: Groups songs with similar audio features
- **Recommendation System**: Suggests similar songs
- **Random Forest**: Predicts song popularity based on features
- **Feature Importance**: Which audio features matter most

### 3. Deep Learning (DL)
- **Neural Network Classifier**: Predicts high-preference songs
- **Autoencoder**: Learns compressed representations of songs
- **Training Visualization**: Loss and accuracy curves

### 4. AI Insights
- **Heatmaps**: Listening activity patterns
- **Trend Analysis**: How your music taste evolves
- **Pattern Recognition**: Identifies listening habits

## Output Files

When you run the Python script, it generates:
- `listening_patterns.png` - Visualization of temporal listening patterns

The Jupyter notebook creates all visualizations inline.

## Data Requirements

The analysis expects two CSV files in the repository root:
1. `My Streaming Activity.csv` - Your streaming history
2. `Scrobble_Features.csv` - Spotify audio features for songs

## Key Features to Explore

### In the Notebook
- **Section 3**: See all your listening statistics and patterns
- **Section 4**: Explore ML clustering and recommendations
- **Section 5**: Deep learning models and predictions
- **Section 6**: AI insights and behavior patterns

### Customization
You can modify parameters in the code:
- Number of clusters (default: 4)
- Neural network architecture
- Number of recommendations
- Visualization styles

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### CUDA Warnings
TensorFlow warnings about CUDA/GPU are normal if you don't have a GPU. The code runs fine on CPU.

### Memory Issues
If you encounter memory issues with large datasets:
- Reduce batch sizes in neural network training
- Use fewer clusters in K-Means
- Sample your data before analysis

## Next Steps

1. Run the analysis to see your listening patterns
2. Explore the visualizations to understand your music taste
3. Use the recommendation system to discover similar songs
4. Modify the code to add your own analyses

## Support

For issues or questions:
- Check the main README.md for detailed documentation
- Review the code comments in the notebook/script
- Open an issue on the GitHub repository

Happy analyzing! 🎵
