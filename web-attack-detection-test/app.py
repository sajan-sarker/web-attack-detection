# app.py - Flask application for ML-Based Intrusion Detection System

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import joblib
import matplotlib
import matplotlib.pyplot as plt
from ensemble import EnsembbleModel
from xai_analyzer import XAIAnalyzer

# Set Matplotlib backend to Agg for non-interactive plotting
matplotlib.use('Agg')

app = Flask(__name__)

# Simulated attack categories
ATTACK_CATEGORIES = {
    0: "Benign",
    1: "Botnet",
    2: "Brute Force",
    3: "DDoS",
    4: "DoS",
    5: "PortScan",
    6: "Web Attack"
}

# Top features for selection
TOP_FEATURES = [
    'Idle Mean', 'Fwd PSH Flags', 'Bwd Packet Length Min', 'Bwd Header Length',
    'Fwd Packets Length Total', 'Active Std', 'Packet Length Variance',
    'Bwd Packets Length Total', 'PSH Flag Count', 'Idle Min', 'Bwd Packet Length Std',
    'Active Max', 'Idle Std', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
    'Fwd Packet Length Min', 'Fwd Packet Length Std', 'Fwd Packet Length Max',
    'Bwd IAT Min', 'FIN Flag Count', 'Bwd Packet Length Mean', 'ACK Flag Count',
    'Avg Packet Size', 'Fwd Header Length', 'Packet Length Max',
    'Fwd Packet Length Mean', 'Fwd IAT Mean', 'Packet Length Min'
]

# Simulated IP addresses
SOURCE_IP = "192.168.1.100"
DESTINATION_IP = "192.168.1.200"

# Ensure the static/images directory exists
os.makedirs('static/images', exist_ok=True)

# Load models and scaler
try:
    model = joblib.load("./em_classifier_v2.pkl")
    scaler = joblib.load("./scaler.pkl")
except Exception as e:
    app.logger.error(f"Error loading model or scaler: {str(e)}")
    raise e

# Reinitialize XAIAnalyzer
xai_analyzer = XAIAnalyzer(model)
dummy_df = pd.DataFrame(np.zeros((1, len(TOP_FEATURES))), columns=TOP_FEATURES)
try:
    xai_analyzer.fit(TOP_FEATURES, ATTACK_CATEGORIES, scaler, dummy_df)
except Exception as e:
    app.logger.error(f"Error initializing XAIAnalyzer: {str(e)}")
    raise e

# Helper function to calculate derived features
def calculate_derived_features(data):
    """Calculate derived features from input data."""
    try:
        if 'Fwd Packets Length Total' in data and 'Fwd Packet Length Max' in data and 'Fwd Packet Length Min' in data:
            fwd_lengths = [data['Fwd Packets Length Total'], data['Fwd Packet Length Max'], data['Fwd Packet Length Min']]
            data['Fwd Packet Length Mean'] = np.mean(fwd_lengths)
            data['Fwd Packet Length Std'] = np.std(fwd_lengths)
        
        if 'Bwd Packets Length Total' in data and 'Bwd Packet Length Max' in data and 'Bwd Packet Length Min' in data:
            bwd_lengths = [data['Bwd Packets Length Total'], data['Bwd Packet Length Max'], data['Bwd Packet Length Min']]
            data['Bwd Packet Length Mean'] = np.mean(bwd_lengths)
            data['Bwd Packet Length Std'] = np.std(bwd_lengths)
        
        if 'Fwd IAT Total' in data and 'Fwd IAT Max' in data and 'Fwd IAT Min' in data:
            fwd_iat = [data['Fwd IAT Total'], data['Fwd IAT Max'], data['Fwd IAT Min']]
            data['Fwd IAT Mean'] = np.mean(fwd_iat)
        
        if 'Packet Length Min' in data and 'Packet Length Max' in data:
            pkt_lengths = [data['Packet Length Min'], data['Packet Length Max']]
            data['Packet Length Mean'] = np.mean(pkt_lengths)
            data['Packet Length Std'] = np.std(pkt_lengths)
            data['Packet Length Variance'] = np.var(pkt_lengths)
            data['Avg Packet Size'] = np.mean(pkt_lengths)
        
        if 'Active Max' in data and 'Active Min' in data:
            active = [data['Active Max'], data['Active Min']]
            data['Active Mean'] = np.mean(active)
            data['Active Std'] = np.std(active)
        
        if 'Idle Min' in data and 'Idle Max' in data:
            idle = [data['Idle Min'], data['Idle Max']]
            data['Idle Mean'] = np.mean(idle)
            data['Idle Std'] = np.std(idle)
        
        return data
    except Exception as e:
        app.logger.error(f"Error calculating derived features: {str(e)}")
        return data

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle CSV file analysis."""
    results = []
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
        app.logger.warning("No CSV file uploaded")
        return jsonify({'error': 'No CSV file uploaded'}), 400
    
    csv_file = request.files['csv_file']
    try:
        df = pd.read_csv(csv_file)
        app.logger.info(f"CSV file loaded successfully with {len(df)} rows")
    except Exception as e:
        app.logger.error(f"Error reading CSV file: {str(e)}")
        return jsonify({'error': 'Error reading CSV file'}), 400
    
    for idx, row in df.iterrows():
        data = row.to_dict()
        app.logger.debug(f"Processing row {idx}: {data}")
        data = calculate_derived_features(data)
        
        # Ensure all TOP_FEATURES are present, fill missing with 0
        features = {}
        for key in TOP_FEATURES:
            features[key] = data.get(key, 0)
        
        app.logger.debug(f"Features for prediction: {features}")
        
        features_array = np.array([list(features.values())])
        try:
            scaled_features = scaler.transform(features_array)
        except Exception as e:
            app.logger.error(f"Error scaling features: {str(e)}")
            continue
        
        try:
            predictions = model.predict(scaled_features)
            multi_class_pred = predictions[0][0]
            attack_category = ATTACK_CATEGORIES.get(multi_class_pred, "Unknown")
        except Exception as e:
            app.logger.error(f"Error making prediction: {str(e)}")
            continue
        
        results.append({
            'protocol': data.get('Protocol', 'Unknown'),
            'source_ip': data.get('Src IP'),
            'destination_ip': data.get('Dst IP'),
            # 'source_ip': SOURCE_IP,
            # 'destination_ip': DESTINATION_IP,
            'time': current_time,
            'activity': attack_category,
            'scaled_features': scaled_features.tolist()
        })
    
    if not results:
        app.logger.warning("No results generated from CSV processing")
        return jsonify({'error': 'No valid data processed from CSV'}), 400
    
    app.logger.info(f"Returning {len(results)} results")
    return jsonify(results)

@app.route('/analyze_xai', methods=['POST'])
def analyze_xai():
    """Generate XAI explanation for the last data point."""
    data = request.json
    try:
        scaled_features = np.array(data['scaled_features'])
        app.logger.debug(f"Scaled features for XAI: {scaled_features}")
        
        # Generate LIME explanation
        xai_analyzer.analyze_lime(scaled_features)
        
        # Save the plot with a timestamp to avoid caching
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        plot_path = f'static/images/lime_explanation_{timestamp}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        app.logger.info(f"XAI plot saved to {plot_path}")
        
        # Verify the file exists
        if not os.path.exists(plot_path):
            app.logger.error("XAI plot file was not created")
            return jsonify({'error': 'Failed to create XAI plot file'}), 500
        
        return jsonify({'plot_url': f'/static/images/lime_explanation_{timestamp}.png'})
    except Exception as e:
        app.logger.error(f"Error generating XAI explanation: {str(e)}")
        return jsonify({'error': f'Error generating XAI explanation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)