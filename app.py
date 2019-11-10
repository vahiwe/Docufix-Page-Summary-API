from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from flask import Flask, request, jsonify, render_template, redirect

# from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk, os
nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_dir)
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# nltk.download('punkt')

LANGUAGE = "english"
SENTENCES_COUNT = 10

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return redirect('https://documenter.getpostman.com/view/9310664/SW18vaAC?version=latest')


@app.route('/', methods=['POST'])
def summarize():
    """ Returns summary of articles """
    text = request.form['text']
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    parser = PlaintextParser.from_string(text,Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    final = []

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        final.append(str(sentence))
    return jsonify(summary=final)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)