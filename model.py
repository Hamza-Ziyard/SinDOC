import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect


#  extractive approach
a=[]
with open('model-files/stopWords.txt', 'r',encoding="utf-16") as f:
    a+=f.readlines()
f.close()
for i in range(0,len(a)):
    a[i]=a[i].rstrip('\n')
stopWords = a


def _create_frequency_table(text_string) -> dict:
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freqTable[wordValue]
                else:
                    sentenceValue[sentence] = freqTable[wordValue]

        if sentence in sentenceValue:
            sentenceValue[sentence] = sentenceValue[sentence] / word_count_in_sentence_except_stop_words

    print(sentenceValue)
    return sentenceValue
    


def _find_average_score(sentenceValue) -> int:

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))
    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence in sentenceValue and sentenceValue[sentence] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def run_summarization(text):
    # Create the word frequency table
    freq_table = _create_frequency_table(text)
    # Tokenize the sentences
    sentences = sent_tokenize(text)
    # : score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)
    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)
    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.0 * threshold)
    return summary



#abstractive approach
#Now we have our model so we can summarise our summary 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Hamza-Ziyard/mT5_multilingual_XLSum-abstaractive-sinhala-news-v2")

model = AutoModelForSeq2SeqLM.from_pretrained("Hamza-Ziyard/mT5_multilingual_XLSum-abstaractive-sinhala-news-v2")


def summarizeText(text, model=model):
    text_encoding = tokenizer(
        text,
        max_length=1000,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    #taking out generated ids according to model saved
    generated_ids = model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )    
    #taking out the predictions
    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    #returning the predictions
    return "".join(preds)



#combined approach
# def hybrid_summarizer(text):
#     extractive = run_summarization(text_str)

#     print('Extractive Summary -+-> ',extractive)

#     abstractive = summarizeText(text)

#     print(' abstractive Summary -+-> ',abstractive)

#     hybrid = summarizeText(extractive)

#     return hybrid  , extractive , abstractive


# hybrid  , extractive , abstractive = hybrid_summarizer(text_str)