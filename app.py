# pip install flask transformers pandas
# pip3 install torch 
# torchvision torchaudio
# labeling ipynb : https://drive.google.com/drive/u/0/folders/16SWta0z2NsFVLwYNLcV1dKukvYnhDxAv
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pandas as pd

app = Flask(__name__)

# Load the sentiment analysis model
classifier = pipeline("sentiment-analysis", model="tegaranggana/f_80_20")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = classifier(text)
        return jsonify(result)
    
@app.route('/predict-file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        # Read the file into a Pandas DataFrame
        df = pd.read_excel(file, engine='openpyxl')
        
        columns = df.columns.tolist()
        data = df.head().to_dict(orient='records')
        
        # Filter out NaN values
        for row in data:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = ""  # or any default value
        
        return jsonify({'columns': columns,'example_data': data}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
