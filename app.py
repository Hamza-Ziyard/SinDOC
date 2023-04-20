import model
from flask import Flask, render_template, request


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


@app.route('/my_endpoint', methods=['POST'])
def my_endpoint():
    text = request.form.get('inputText')
    # do something with the text data
    print(text)
    
    extractive = model.run_summarization(text)
    print('Extractive Summary -+-> ',extractive)
    
    abstractive = model.summarizeText(text)
    print(' abstractive Summary -+-> ',abstractive)
    hybrid = model.summarizeText(extractive)
    print(hybrid)

    return hybrid
    # return hybrid  , extractive , abstractive
    # return "Received text: "



if __name__ == '__main__':
    app.run(debug=True)
