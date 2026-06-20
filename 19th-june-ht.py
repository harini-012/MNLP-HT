'''Home Task
Date: 19-06-2026
1. A company is building a Resume Screening System. The system receives raw resume text and must 
preprocess it to extract meaningful keywords for ranking candidates. You are required to build an 
NLP pipeline that performs the following steps: Tasks: 1. Tokenize the input text into words 2. 
Remove stopwords 3. Perform stemming 4. Perform lemmatization 5. Apply POS tagging 6. Compute 
frequency distribution of remaining words'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import pos_tag,FreqDist
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
resume_text="""
John Doe is a software engineer with 5 years of experience in Python,
Machine Learning, Data Analysis, SQL, and Web Development.
He has worked on AI projects and developed scalable applications.
"""
tokens=word_tokenize(resume_text)
print("Tokens:")
print(tokens)
stop_words=set(stopwords.words('english'))
filtered_tokens=[word.lower() for word in tokens if word.isalnum() and word.lower() not in 
stop_words]
print("\nAfter Stopword Removal:")
print(filtered_tokens)
stemmer=PorterStemmer()
stemmed_words=[stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Words:")
print(stemmed_words)
lemmatizer=WordNetLemmatizer()
lemmatized_words=[lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nLemmatized Words:")
print(lemmatized_words)
pos_tags=pos_tag(lemmatized_words)
print("\nPOS Tags:")
print(pos_tags)
freq_dist=FreqDist(lemmatized_words)
print("\nFrequency Distribution:")
for word,freq in freq_dist.items():
 print(word,":",freq
