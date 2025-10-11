#!/usr/bin/env python3
"""
Streaming Activity Analysis Script
Performs EDA, ML, and DL analysis on streaming activity data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load streaming activity and features datasets"""
    print("Loading datasets...")
    streaming_df = pd.read_csv('My Streaming Activity.csv')
    features_df = pd.read_csv('Scrobble_Features.csv')
    
    print(f"Streaming Activity Shape: {streaming_df.shape}")
    print(f"Features Shape: {features_df.shape}")
    
    return streaming_df, features_df


def preprocess_data(streaming_df):
    """Preprocess streaming data and extract temporal features"""
    print("\nPreprocessing data...")
    
    # Convert timestamps
    streaming_df['TimeStamp_UTC'] = pd.to_datetime(streaming_df['TimeStamp_UTC'])
    streaming_df['TimeStamp_Central'] = pd.to_datetime(streaming_df['TimeStamp_Central'])
    
    # Extract temporal features
    streaming_df['Year'] = streaming_df['TimeStamp_UTC'].dt.year
    streaming_df['Month'] = streaming_df['TimeStamp_UTC'].dt.month
    streaming_df['Day'] = streaming_df['TimeStamp_UTC'].dt.day
    streaming_df['Hour'] = streaming_df['TimeStamp_UTC'].dt.hour
    streaming_df['DayOfWeek'] = streaming_df['TimeStamp_UTC'].dt.dayofweek
    streaming_df['WeekOfYear'] = streaming_df['TimeStamp_UTC'].dt.isocalendar().week
    
    print("Temporal features extracted successfully!")
    
    return streaming_df


def perform_eda(streaming_df, features_df):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\nTotal songs played: {len(streaming_df):,}")
    print(f"Unique songs: {streaming_df['Song'].nunique():,}")
    print(f"Unique performers: {streaming_df['Performer'].nunique():,}")
    print(f"Unique albums: {streaming_df['Album'].nunique():,}")
    
    date_range = (streaming_df['TimeStamp_UTC'].max() - streaming_df['TimeStamp_UTC'].min()).days
    print(f"Date range: {streaming_df['TimeStamp_UTC'].min()} to {streaming_df['TimeStamp_UTC'].max()}")
    print(f"Total days: {date_range}")
    
    # Top artists and songs
    print("\nTop 5 Artists:")
    print(streaming_df['Performer'].value_counts().head())
    
    print("\nTop 5 Songs:")
    print(streaming_df['Song'].value_counts().head())
    
    # Merge datasets
    merged_df = streaming_df.merge(
        features_df, 
        on=['Performer', 'Song'], 
        how='left',
        suffixes=('_streaming', '_features')
    )
    
    return merged_df


def visualize_patterns(streaming_df):
    """Create visualizations for temporal patterns"""
    print("\nGenerating visualizations...")
    
    # Listening patterns by hour
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    hourly_counts = streaming_df['Hour'].value_counts().sort_index()
    plt.bar(hourly_counts.index, hourly_counts.values, color='skyblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Songs')
    plt.title('Listening Activity by Hour of Day')
    plt.xticks(range(0, 24))
    
    # Day of week
    plt.subplot(1, 2, 2)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = streaming_df['DayOfWeek'].value_counts().sort_index()
    plt.bar(range(7), dow_counts.values, color='lightcoral')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Songs')
    plt.title('Listening Activity by Day of Week')
    plt.xticks(range(7), day_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('listening_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: listening_patterns.png")
    plt.close()


def perform_clustering(merged_df):
    """Perform K-Means clustering on audio features"""
    print("\n" + "="*60)
    print("MACHINE LEARNING: CLUSTERING")
    print("="*60)
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
    features_available = merged_df.dropna(subset=audio_features)
    
    if len(features_available) < 100:
        print("Insufficient data for clustering")
        return None
    
    X_cluster = features_available[audio_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K-Means clustering
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    features_available['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"\nClustered {len(features_available)} songs into {optimal_k} groups")
    
    # Cluster characteristics
    for cluster in range(optimal_k):
        cluster_data = features_available[features_available['Cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_data)} songs):")
        print(cluster_data[audio_features].mean())
    
    return features_available


def train_ml_model(streaming_df, features_df):
    """Train Random Forest model for play count prediction"""
    print("\n" + "="*60)
    print("MACHINE LEARNING: PREDICTION MODEL")
    print("="*60)
    
    # Create play count feature
    song_popularity = streaming_df.groupby('Song').size().reset_index(name='PlayCount')
    ml_data = features_df.merge(song_popularity, on='Song', how='inner')
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness']
    ml_data = ml_data.dropna(subset=audio_features + ['PlayCount'])
    
    if len(ml_data) < 100:
        print("Insufficient data for ML model")
        return None
    
    X = ml_data[audio_features].values
    y = ml_data['PlayCount'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRandom Forest Model Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    return rf_model, mae, r2


def train_deep_learning_model(streaming_df, features_df):
    """Train neural network for song preference prediction"""
    print("\n" + "="*60)
    print("DEEP LEARNING: NEURAL NETWORK")
    print("="*60)
    
    song_popularity = streaming_df.groupby('Song').size().reset_index(name='PlayCount')
    ml_data = features_df.merge(song_popularity, on='Song', how='inner')
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness']
    ml_data = ml_data.dropna(subset=audio_features + ['PlayCount'])
    
    if len(ml_data) < 100:
        print("Insufficient data for DL model")
        return None
    
    # Create binary target
    threshold = ml_data['PlayCount'].quantile(0.75)
    ml_data['HighPreference'] = (ml_data['PlayCount'] >= threshold).astype(int)
    
    X_dl = ml_data[audio_features].values
    y_dl = ml_data['HighPreference'].values
    
    scaler_dl = MinMaxScaler()
    X_dl_scaled = scaler_dl.fit_transform(X_dl)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_dl_scaled, y_dl, test_size=0.2, random_state=42
    )
    
    # Build model
    model = models.Sequential([
        layers.Input(shape=(len(audio_features),)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nNeural Network Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    return model, test_accuracy, test_auc


def generate_summary(streaming_df, ml_results=None, dl_results=None):
    """Generate comprehensive summary of analysis"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total songs played: {len(streaming_df):,}")
    print(f"  • Unique songs: {streaming_df['Song'].nunique():,}")
    print(f"  • Unique artists: {streaming_df['Performer'].nunique():,}")
    
    date_range = (streaming_df['TimeStamp_UTC'].max() - streaming_df['TimeStamp_UTC'].min()).days
    print(f"  • Time period: {date_range} days")
    
    print(f"\n🎵 Top Artist: {streaming_df['Performer'].value_counts().index[0]}")
    print(f"  • Plays: {streaming_df['Performer'].value_counts().values[0]}")
    
    print(f"\n🎶 Top Song: {streaming_df['Song'].value_counts().index[0]}")
    print(f"  • Plays: {streaming_df['Song'].value_counts().values[0]}")
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"\n⏰ Listening Patterns:")
    print(f"  • Peak hour: {streaming_df['Hour'].mode()[0]}:00")
    print(f"  • Peak day: {day_names[streaming_df['DayOfWeek'].mode()[0]]}")
    
    avg_per_day = len(streaming_df) / date_range if date_range > 0 else 0
    print(f"  • Average songs per day: {avg_per_day:.1f}")
    
    if ml_results:
        print(f"\n🤖 ML Model Performance:")
        print(f"  • Random Forest MAE: {ml_results[1]:.2f}")
        print(f"  • Random Forest R²: {ml_results[2]:.3f}")
    
    if dl_results:
        print(f"\n🧠 DL Model Performance:")
        print(f"  • Neural Network Accuracy: {dl_results[1]:.3f}")
        print(f"  • Neural Network AUC: {dl_results[2]:.3f}")
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)


def main():
    """Main execution function"""
    print("="*60)
    print("STREAMING ACTIVITY EXPLORATION")
    print("EDA, AI, ML, and DL Analysis")
    print("="*60)
    
    # Load and preprocess data
    streaming_df, features_df = load_data()
    streaming_df = preprocess_data(streaming_df)
    
    # Perform EDA
    merged_df = perform_eda(streaming_df, features_df)
    
    # Generate visualizations
    visualize_patterns(streaming_df)
    
    # Perform clustering
    clustered_data = perform_clustering(merged_df)
    
    # Train ML model
    ml_results = train_ml_model(streaming_df, features_df)
    
    # Train DL model
    dl_results = train_deep_learning_model(streaming_df, features_df)
    
    # Generate summary
    generate_summary(streaming_df, ml_results, dl_results)


if __name__ == "__main__":
    main()
