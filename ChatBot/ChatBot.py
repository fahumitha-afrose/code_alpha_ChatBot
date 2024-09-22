import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('ChatBot.txt', 'r', errors='ignore')
raw_doc = f.read().lower()
nltk.download('punkt')
nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Greet_inputs = ("hello", "hi", "greetings", "what's up", "hey")
Greet_responses = ["hi", "hey", "hello", "hi there", "i am glad! you are talking with me"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in Greet_inputs:
            return random.choice(Greet_responses)

def response(user_response):
    robo1_response = ''
    sent_tokens.append(user_response)
    
    Tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = Tfidfvec.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  
    idx = vals.argsort()[0][-1]  
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    
    if req_tfidf == 0:
        robo1_response = "I am sorry! I don't understand you."
    else:
        robo1_response = sent_tokens[idx]
    
    sent_tokens.remove(user_response)  
    return robo1_response

flag = True
print("BOT: My name is Farose. Let's have a conversation! Type 'bye' to exit.")
while flag:
    user_response = input().lower()
    if user_response != 'bye':
        if user_response in ['thanks', 'thank you']:
            flag = False
            print("BOT: You are welcome...")
        else:
            if greet(user_response) is not None:
                print("BOT: " + greet(user_response))
            else:
                print("BOT: ", end="")
                print(response(user_response))
    else:
        flag = False
        print("BOT: Goodbye! Take care...")
