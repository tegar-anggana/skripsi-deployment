# pip install flask transformers
# pip3 install torch 
# torchvision torchaudio
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

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

if __name__ == '__main__':
    app.run(debug=True)
