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
    
    extractive = model.run_summarization(text)
    print('Extractive Summary -+-> ',extractive)
    
    return extractive

@app.route('/my_endpoint_abstract', methods=['POST'])
def my_endpoint_abstract():
    text = request.form.get('inputText')
    print(text)
    
    abstractive = model.summarizeText(text)
    print(' abstractive Summary -+-> ',abstractive)

    return abstractive

@app.route('/my_endpoint_combined', methods=['POST'])
def my_endpoint_combined():
    text = request.form.get('inputText')
    print(text)
    
    extractive = model.run_summarization(text)
    print('Extractive Summary -+-> ',extractive)
    
    # abstractive = model.summarizeText(text)
    # print(' abstractive Summary -+-> ',abstractive)
    hybrid = model.summarizeText(extractive)
    print(hybrid)

    return hybrid



if __name__ == '__main__':
    app.run(debug=True)
