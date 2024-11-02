
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator


translator = Translator()
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    return text_content

def detect_language(text):
    return detect(text)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def find_best_answer(question, sentences):
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    best_match_index = similarity.argmax()
    return sentences[best_match_index]

def translate_text(text, src_lang, target_lang):
    return translator.translate(text, src=src_lang, dest=target_lang).text

def chatbot_interaction(url):
    ## Main
    text_content = scrape_content(url)
    content_language = detect_language(text_content)
    sentences = preprocess_text(text_content)
    
    print(f"Content detected in language: {content_language}")
    print("Welcome to the URL-based Q&A chatbot! Ask me anything about the content at the URL provided.")
    print("Type 'exit' to end the conversation.")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Chatbot: Thank you for chatting. Goodbye!")
            break

        question_language = detect_language(question)
        
        if question_language != content_language:
            question = translate_text(question, src_lang=question_language, target_lang=content_language)

        answer = find_best_answer(question, sentences)

        if question_language != content_language:
            answer = translate_text(answer, src_lang=content_language, target_lang=question_language)

        print("Chatbot:", answer)

if __name__ == "__main__":
    example_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"  # Replace with the URL
    chatbot_interaction(example_url)

