import model
from flask import Flask, render_template, request,jsonify


app = Flask(__name__)

app.config['DEBUG'] = True                      # Turn off for production
app.config['TEMPLATES_AUTO_RELOAD'] = True      # Turn off for production

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app')
def summarizer():
    return render_template('app.html')


@app.route('/navbar')
def navbar():
    return render_template('navbar.html')


@app.route('/my_endpoint_extract', methods=['POST'])
def my_endpoint_extract():
    text = request.form.get('inputText')
    print(text)
    
    extractive_summary = model.extractive_summarizer(text)
    print('Extractive Summary -+-> ',extractive_summary)
    
    return extractive_summary

@app.route('/my_endpoint_abstract', methods=['POST'])
def my_endpoint_abstract():
    text = request.form.get('inputText')
    print(text)
    
    abstractive_summary = model.abstractive_summarizer(text)
    print(' abstractive Summary -+-> ',abstractive_summary)

    return abstractive_summary

@app.route('/my_endpoint_combined', methods=['POST'])
def my_endpoint_combined():
    text = request.form.get('inputText')
    print(text)
    
    extractive_summary = model.extractive_summarizer(text)
    
    combined_summary = model.abstractive_summarizer(extractive_summary)
    print('Combined Summary -+-> ',combined_summary)

    return combined_summary



if __name__ == '__main__':
    app.run(debug=True)
