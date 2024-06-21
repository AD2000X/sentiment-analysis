# -*- coding: utf-8 -*-
# The sentiment differences between the writing styles of Jane Austen and Charles Dickens - Part 2

# 1. Introduction, objectives and background

## 1.1 Introduction

In part 1, we explored the overall sentiment in the random shuffle literature works dataset of Jane Austen and of Charles Dickens. We concluded that the literature works of Jane Austen have a higher overall score in positive sentiment, the literature works of Charles Dickens have a higher fluctuation in negative sentiment, and the literature works of Charles Dickens are higher in neutral sentiment.

In DSM020, we are going to use the same pre-processed datasets of both two authors to further explore the different segments, including Syntactic Analysis, Part-of-Speech, and Sentiment Analysis with two more indexes - compound sentiment and subjective. Furthermore, we are going to explore the technique of Term Frequency - Inverse Document Frequency (TF-IDF), Named Entity Recognition (NER), Sentiment Analysis by Characters, and Semantic Search by BERT.

## 1.2 Objective
◆ Discover the syntactic usage tendency of the two authors.

◆ Investigate the usage amount in adjectives, adverbs, and verbs of two authors.

◆ The sentiment beyond positive, negative and neutral - compound sentiment in sentences.

◆ The polarity and sujectivity of the two authors.

◆ Preliminary examination about TF-IDF.

◆ Apply NER to the sentiment score of all characters in both authors' works.

## 1.3 Methodology (for DSM020)

### Dataset
Because of the limitation of the submission zip file size of coursework 1, the dataset of the paragraph collection of Jane Austen and of Charles Dickens has been randomly decreased to 2,000 lines in coursework 1. The way we chose randomly decreased lines is due to the dataset being a collection of paragraphs of their works and we aim to analyze the sentiment orientation in writing instead of in a certain piece of work. Therefore, we are going to use the same dataset to avoid inconsistency and ambiguity in the analysis results in DSM002.

### Workflow
Since we have an overview of their sentiment orientation, we want to further examine from different angles of the sentiment in their works. Firstly, we use the dataset from our coursework 1 to look into the sentence structure. Second, we take a deep look at the part of speech among adjectives, adverbs, and verbs. Afterwards, we mainly inspect the compound sentiments in the senteces. Additionally, we take a short view on the overall polarity and subjectively of their works. Next, we take three simple expiriments coding toward to TF-IDF, NER and BERT to apply on the datasets.

### Technical improvements since CW 1
Overall, in coursework 1, we simply use VADER and wordcloud to process sentiment analysis in coursework 1, but there should be more applications in NLP for sentiment analysis for our exploration. The specificity of human language is that it is structured somehow but not 100%, so it can be applied to different situations in different ways by different people. Therefore, we decided to start from a linguistics perspective to examine the sentiment in their works. For sentence structure analysis, we use SpaCy and further visualize in a dependency tree, which is a new attempt for us. Moreover, we further examine the POS in word cloud, Polarity and Subjective for two authors, sentiment analysis in compound sentiment, simply review the TF-IDF technique, and NER for characters' sentiment analysis. Those approaches are for better preparation for our NLP module in Term 2.

Furthermore, we use "os.path.dirname(os.path.abspath('file'))" to make sure that our codes can successfully be executed on different computers, but we change to Google Colab for simply mount Google Drive as a new attempt. For stop words, we removed stop words at the very beginning in coursework 1, but the reason we did not proceed with the same step is that we thought that the stop words in literature work possibly affect the sentiment of a sentence and we aim to examine the syntactic and POS, and we also want to check if the removal of stop words would affect the sentiment analysis in literature works. Hence, we choose to remove stop words before we design the codes of TF-IDF and NER. Subsequently, in order to complete sentiment analysis by authors of their subjective, we try Textblob, which is also a tool for sentiment analysis, instead of VADER. Moreover, during remove stop word, we also introduce "Stemmer" to prepare for TF-IDF and NER. Lastly, we try to move all libraries and modules to the beginning of the program to improve the readability of the program.

### Ethics
Moreover, to better complete the rubric from coursework 1 "Ethics of use of data have been considered", we are going to further examine the ethics use. Jane Austen(1775 - 1817) and Charles Dickens(1812 - 1870) are both authors of the United Kingdom. From the legal aspect, according to the Copyright, Designs and Patents Act 1988, Section 12, Section 15, and Section 28 -31, both authors' work are open to public without limitation. From the cultral aspect, both authors' work has been adapted in different artistic works, and their's no evidense show that a single indepent, a group or race has reported offensed. Additionally, from a biological perspective, both of their works are 153 years ago, which means under the common knowledge of human being's life cyclem, it is barely a chance that a living human was alluded. Last bbut not least, the dataset on Kaggle has been marked as CC0(Public Domain). Therefore, we can conclude that the dataset is safe to use in our coursework.

# Project Limitations
#### Considering the rigorous preprocess dataset in each step of NLP, coursework 2 could be implemented with a more robust procedure. We tried to introduce LDA and LSA to enhance the overview of our sentiment analysis landscape, but maybe there's some missing in the preprocess of datasets, the results we had are full of conjunctions and prepositions. It is not helpful and not enough to build a clear picture of topics - even if we enlarged our topic numbers to 100. Hence we have to drop this technique in our coursework 2. Or, perhaps in literature sentiment analysis, we should take machine learning way instead of traditional statistics way. Moreover, NLP missions in a certain field, such as literature works analysis, could require related specialists equipped with certain domain knowledge to generate more insights. In addition, we simply examine the concept of TF-IDF without building further sentiment analysis applications. Also, we have researched the concept of BERT and built a sementic search engine by following a reference book, but we could not generate any useful sentiment analysis by just that engine, so we drop it from our coursework 2. Lastly, codes could be organised in a better manner, such as remove some duplicate codes in different sections.

# 2. Preparation for Dataset Analysis

## 2.1 Install all required libraries
"""

!pip install spacy  # Syntactic Analysis(Parsing)
!pip install nltk
!pip install numpy  #Radar Chart
!pip install matplotlib
!pip install textblob # Subjective
!pip install wordcloud
!pip install scikit-learn # TF-IDF

"""## 2.1 Import all required modules"""

# Syntactic Analysis(Parsing)
import spacy
from collections import Counter
# Dependency Tree
# import spacy
from spacy import displacy

# Part-of-Speech(POS)
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
# Wordcloud
from wordcloud import WordCloud

# Sentiment Analysis - by Authors
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
# Sentiment Analysis - by Authors, Radar Chart
import numpy as np
import matplotlib.pyplot as plt
# Sentiment Analysis - by Authors, Polarity and Subjective
from textblob import TextBlob

# Stop Word Removal
# import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Term Frequency - Inverse Document Frequency (TF-IDF), TfidfVectorizer
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Named Entity Recognition (NER)
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Sentiment analysis - by Characters(example)
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import pandas as pd

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Indicate the path to the file in Google Drive
file_name_austen = '/content/drive/My Drive/austen.txt'  # your own route
file_name_dickens = '/content/drive/My Drive/dickens.txt'  # your own route

# Initialize the dictionary used to store text for each author
authors_texts = {
    'Jane Austen': '',
    'Charles Dickens': ''
}

# Read and store text
with open(file_name_austen, 'r', encoding='utf-8') as file:
    authors_texts['Jane Austen'] = file.read()

with open(file_name_dickens, 'r', encoding='utf-8') as file:
    authors_texts['Charles Dickens'] = file.read()

"""# 2.2 Syntactic Analysis(Parsing)

#### We are interested in the syntactic so we decided to use en_core_web_sm from SpaCy to analyse the sentence structures. The key step here is using "nlp(text)"(including tokenized, POS, dependency) and "doc.sents" to iterative the sentences in the text. The results show that Jane Austen used more interjections and complete sentence structures, while Charles Dickens more used verbs and proper nouns, meaning that Jane Austen may be more likely to use adverbs to describe actions or emotions in detail, while Charles Dickens may use more nouns and interjections to enhance the vividness and expressiveness of narratives.
"""

!python -m spacy download en_core_web_sm

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Sentence structure analysis for both author's text
for author, text in authors_texts.items():
    # NLP process
    doc = nlp(text)

    # Syntactic Analysis
    sentence_structures = []
    for sent in doc.sents:
        structure = []
        for token in sent:
            structure.append(token.pos_)
        structure = " ".join(structure)
        sentence_structures.append(structure)

    # Statistics of the most common sentence structures
    structure_freq = Counter(sentence_structures)
    most_common_structures = structure_freq.most_common(10)
    print(f"Author: {author}")
    print("Most common sentence structures:")
    for structure, freq in most_common_structures:
        print(f'{structure}: {freq}')
    print("\n")

"""# 2.2 Dependency Tree
#### We visualized the dependency tree so that we could simply observe their writing orientation in the PUNCT structure.
"""

# Perform sentence structure analysis on both authors' text and generate a syntactic dependency tree
for author, text in authors_texts.items():
    # NLP processing
    doc = nlp(text)

    # Chooese the first one sentence as example
    example_sentences = list(doc.sents)[:1]

    print(f"Author: {author}")
    print("Example Sentence Structures:")

    for sent in example_sentences:
        displacy.render(sent, style='dep', jupyter=True, options={'distance': 90})

"""# 2.3 Part-of-Speech(POS)
#### A standard sentence is constructed by Subject+Verb+Objective. The reason we want to take a view of adjectives, adverbs, and verbs is that these kinds of words usually carry more and various sentiments. The result shows that Jane Austen's text has a greater number of adjectives, adverbs, and verbs than Charles Dickens's text, which fits our result in CW 1, namely Jane Austen's text has a higher score of positive sentiment and a lower score of neutral sentiment. However, Charles Dickens has a higher fluctuation in negative sentiment score, we can further refer that the words used in his text are more acute.
"""

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Define a function to label adjectives, adverbs and verbs
def tag_words(text):
    words = word_tokenize(text)  # tokenize
    tagged_words = pos_tag(words)  # POS tagging

    # Filter adjectives, adverbs and verbs
    adjectives = [word for word, tag in tagged_words if tag in ('JJ', 'JJR', 'JJS')]
    adverbs = [word for word, tag in tagged_words if tag in ('RB', 'RBR', 'RBS')]
    verbs = [word for word, tag in tagged_words if tag.startswith('VB')]
    return adjectives, adverbs, verbs

# Initialize the dictionary used to store text for two authors
authors_texts = {
    'Jane Austen': '',
    'Charles Dickens': ''
}

file_name_austen = '/content/drive/My Drive/austen.txt'
file_name_dickens = '/content/drive/My Drive/dickens.txt'

with open(file_name_austen, 'r', encoding='utf-8') as file:
    authors_texts['Jane Austen'] = file.read()

with open(file_name_dickens, 'r', encoding='utf-8') as file:
    authors_texts['Charles Dickens'] = file.read()

# Initialize a dictionary to store adjective, adverb, and verb counts for two authors
adjectives_count = {'Jane Austen': Counter(), 'Charles Dickens': Counter()}
adverbs_count = {'Jane Austen': Counter(), 'Charles Dickens': Counter()}
verbs_count = {'Jane Austen': Counter(), 'Charles Dickens': Counter()}

# Tagging adjectives, adverbs and verbs in two authors' text
for author, text in authors_texts.items():
    adjectives, adverbs, verbs = tag_words(text)
    adjectives_count[author].update(adjectives)
    adverbs_count[author].update(adverbs)
    verbs_count[author].update(verbs)

# Print counts for two authors
for author in authors_texts:
    print(f"Author: {author}")
    print("Adjectives Count:", sum(adjectives_count[author].values()))
    print("Adverbs Count:", sum(adverbs_count[author].values()))
    print("Verbs Count:", sum(verbs_count[author].values()))
    print("\n")

# Calculate the total words for two authors
authors_counts = {
    author: [
        sum(adjectives_count[author].values()),
        sum(adverbs_count[author].values()),
        sum(verbs_count[author].values())
    ]
    for author in authors_texts
}

# create data for histogram
labels = ['Adjectives', 'Adverbs', 'Verbs']
authors = list(authors_texts.keys())
counts_austen = authors_counts['Jane Austen']
counts_dickens = authors_counts['Charles Dickens']

x = np.arange(len(labels))
width = 0.35

# create histogram
plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, counts_austen, width, label='Jane Austen')
rects2 = plt.bar(x + width/2, counts_dickens, width, label='Charles Dickens')

# Add text labels, titles, and custom X-axis ticks
plt.ylabel('Counts')
plt.title('Word Usage Comparison between Jane Austen and Charles Dickens')
plt.xticks(x, labels)
plt.legend()

plt.show()

"""# 2.4 Part-of-Speech(Tagging) - Wordcloud
#### In our coursework 1, we summarise all words in 2 word cloud, which is slightly ambiguos, so we try to visualise words by the POS. An interesting thing in the adjective word cloud of Jane Austen, "ill" is a high-frequency word, so we can speculate that many characters are in an "ill" situation in her works. In the adverb word cloud, "yet" appears in both their word cloud, consider The use of "yet" usually depends on the tense and context of the sentence, and it can play different roles in the sentence, thereby changing the overall meaning of the sentence. Surprisingly, the word "love" highly appears in Charles Dickens's text, but not in Jane Austen's word cloud, perhaps it is because "Jane Austen usually writes more in a humorous way"?
"""

# # Tagging adjectives, adverbs and verbs in two authors' text
# for author, text in authors_texts.items():
#     adjectives, adverbs, verbs = tag_words(text)
#     adjectives_count[author].update(adjectives)
#     adverbs_count[author].update(adverbs)
#     verbs_count[author].update(verbs)

# Create separate word clouds for adjectives, adverbs, and verbs
word_types = ['Adjectives', 'Adverbs', 'Verbs']
word_counts = [adjectives_count, adverbs_count, verbs_count]

for i, word_type in enumerate(word_types):
    plt.figure(figsize=(16, 8))

    for j, author in enumerate(authors_texts):
        # Combine specific types of vocabulary
        words = ' '.join(word_counts[i][author].keys())
        wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(words)

        # disply wordcloud
        plt.subplot(1, 2, j+1)  # create subgraph
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{word_type} Word Cloud for {author}")

    plt.suptitle(f'Comparison of {word_type} between Jane Austen and Charles Dickens')
    plt.show()

"""# 2.5 Sentiment Analysis - by Authors
#### The main point here is to check the compound sentiment score for sentences. The result shows that the score of Jane Austen's text is much higher than Charles Dickens', which may indicate the differences in their writing style.
"""

nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function for sentiment analysis
def analyze_sentiment_vader(text):
    return analyzer.polarity_scores(text)

# Characters list
characters = ["Elizabeth", "Darcy", "Oliver", "Oliver Twist", "Bill", "Bill Sikes"]

# Initialize an empty DataFrame to store sentiment analysis results
columns = ['Character', 'Positive', 'Negative', 'Neutral', 'Compound']
character_sentiments = {character: {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Compound': 0, 'Count': 0} for character in characters}

# Sentiment analysis two authors' text
for author, text in authors_texts.items():
    lines = text.split('\n')
    for line in lines:
        for character in characters:
            if character in line:
                sentiment_scores = analyze_sentiment_vader(line)
                # Accumulate each sentiment score
                character_sentiments[character]['Positive'] += sentiment_scores['pos']
                character_sentiments[character]['Negative'] += sentiment_scores['neg']
                character_sentiments[character]['Neutral'] += sentiment_scores['neu']
                character_sentiments[character]['Compound'] += sentiment_scores['compound']
                # Record the number of sentences to calculate the average score
                character_sentiments[character]['Count'] += 1

df_list = []

for character, scores in character_sentiments.items():
    if scores['Count'] > 0:  # Avoid dividing by zero, tricky part
        averages = {score: total/scores['Count'] for score, total in scores.items() if score != 'Count'}
        df_list.append(pd.DataFrame([averages], index=[character]))

# Merge all DataFrames
character_sentiment_averages = pd.concat(df_list)

# Print the average sentiment score for each character
print(character_sentiment_averages)

"""# 2.6 Sentiment Analysis - by Authors, Radar Chart
#### This part serves as the visualization for 2.5.
"""

# Create radar chart angles and labels
labels = np.array(['Positive', 'Negative', 'Neutral', 'Compound'])
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close radar chart

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Loop through plotting sentiment data for each author
for author, values in sentiments.items():
    stats = [values[label] for label in labels]
    stats += stats[:1]  # close score tricky part
    ax.plot(angles, stats, label=author)
    ax.fill(angles, stats, alpha=0.25)

# Set parameters of radar chart
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)  #  Matched the label and number of angles

plt.title("Sentiment Analysis Radar Chart")
plt.legend(loc='upper right')
plt.show()

"""# 2.7 Sentiment Analysis - by Authors, Polirity and Subjective
#### Both authors' works display a degree of positive emotion, meaning they may reflect optimistic themes in their work eventually. Regarding subjectivity, the scores of both authors' works display a mixture of objective and subjective parts.
"""

# Use the existing text data
corpus_df = pd.DataFrame.from_dict(authors_texts, orient='index').reset_index()
corpus_df.columns = ['Author', 'Text']

# Define sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

# Sentiment analysis on each text
corpus_df['Sentiment'] = corpus_df['Text'].apply(analyze_sentiment)

# Display sentiment analysis results
for index, row in corpus_df.iterrows():
    print(f"Author: {row['Author']}")
    print(f"Sentiment: Polarity={row['Sentiment'].polarity}, Subjectivity={row['Sentiment'].subjectivity}\n")

"""# 2.8 Preprocess Stop Word for TF-IDF and NER.
#### Even though it is not usually adapted in NLP, we choose to stemmer all words in the text and then remove them all for a consistency purpose because for our first attmept for TF-IDF. We print out the results for simply examine the stopwords.
"""

nltk.download('stopwords')
nltk.download('punkt')  # dataset for tokenize

def process_text_file(file_path, author):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        # Remove stop words and tokenize
        text = text.replace("\n", " ")
        text = re.sub(r'[^\w\s\']', '', text)
        words = word_tokenize(text)

        # Initialize the stemmer
        stemmer = PorterStemmer()

        # Get stop words
        stopwords_list = stopwords.words('english')

        # Extract stem stop words from text
        stopwords_in_text = [stemmer.stem(word.lower()) for word in words if word.lower() in stopwords_list][:100]

        # Remove stop words from the text and stem the remaining words
        filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stopwords_list]

        return {
            'Author': author,
            'Text': ' '.join(filtered_words),
            'Words': filtered_words,
            'RemovedStopwords': stopwords_in_text
        }

# File path
file_name_austen = '/content/drive/My Drive/austen.txt'
file_name_dickens = '/content/drive/My Drive/dickens.txt'

# Process two authors' text and collect data
authors = ['Jane Austen', 'Charles Dickens']
data_list = []

for file_name, author in zip([file_name_austen, file_name_dickens], authors):
    data = process_text_file(file_name, author)
    data_list.append(data)

# Create DataFrame
corpus_df = pd.DataFrame(data_list)

# Print DataFrame
print(corpus_df)

# Print the first 100 stop words removed for each author
for author_data in data_list:
    print(f"{author_data['Author']}'s Top 100 Removed Stopwords:")
    print(author_data['RemovedStopwords'])
    print("\n")

"""# 2.9 Term Frequency - Inverse Document Frequency (TF-IDF)
#### Even though it is not usually adapted in NLP, we choose to stemmer all words in the text and then remove them all for a consistency purpose because for our first attempt for TF-IDF. The concept of TF-IDF is that we calculate the Term Frequency in the dataset, and then we calculate the Inverse Document Frequency of each word in the dataset. The ratio of them represents the importance of a word in the dataset. For each dataset, a vector is constructed containing the TF-IDF scores for all words so that each dataset is converted into a vector, which their dimensions are equal to the total number of words in the datasets.
"""

# Initialize the TF-IDF vectorizer and adjust the min_df and max_df parameters
# min_df = 1 means the word appears in at least 1 dataset
# max_df = 0.95 means the word appears in at most 95% of the datasets
vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)

# Extract processed text
texts = corpus_df['Text'].tolist()

# Fit and transform text using the TF-IDF vectorizer
tfidf_matrix = vectorizer.fit_transform(texts)

# Get feature name (word)
feature_names = vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame
# Each row represents a text, each column represents a word in the vocabulary,
#  and each element is the TF-IDF score of the corresponding word in the text.
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=corpus_df['Author'])

# Check TF-IDF DataFrame
print(df_tfidf)

# Print out feature names (words)
print(vectorizer.get_feature_names_out())

"""# 2.10 Named Entity Recognition (NER)
#### At first, we tried to proceed with NER for listing all the characters' names to store in a dataframe and calculated all the characters' sentiment scores so that we could display all characters' sentiment colors in a 3D sphere. However, it means we had to calculate 6,119 times(names) in separate nearly 12,000 lines two datasets, and it is way out of our computing power (sadly, the program ran for 3 hours without output). Therefore, we choose to examine certain famous characters' sentiment scores, providing a different approach for executing sentiment analysis in literature works in our DSM020 coursework.
"""

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def extract_person_names(text):
    # tokenize
    words = word_tokenize(text)
    # POS taging
    pos_tags = pos_tag(words)
    # Named entity recognition by using ne_chunk
    named_entities = ne_chunk(pos_tags)

    # Extract person name entity
    person_names = []
    for entity in named_entities:
        if isinstance(entity, Tree) and entity.label() == 'PERSON':
            name = " ".join([leaf[0] for leaf in entity.leaves()])
            person_names.append(name)
    return person_names

# Use the text in the authors_texts dictionary for name entity extraction
for author, text in authors_texts.items():
    person_names = extract_person_names(text)
    print(f"Author: {author}")
    print("Characters Names:")
    print(person_names)
    print("Total Number of Character Names:", len(person_names))
    print("\n")

"""# 2.11 Sentiment Analysis - by Characters(example)
#### We pick famous characters "Elizabeth" and "Darcy" in Jane Austen's work, and pick "Oliver", "Oliver Twist", "Bill", and "Bill Sikes" in Charles Dickens's work. The interesting thing here is that there are significant differences in compound sentiment scores when mentioning "Oliver" and "Oliver Twist", we can assume that sentences with "Oliver" are more positive yet "Oliver Twist" is with way more negative sentiment.
"""

nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function for sentiment analysis
def analyze_sentiment_vader(text):
    return analyzer.polarity_scores(text)

# Character list
characters = ["Elizabeth", "Darcy", "Oliver", "Oliver Twist", "Bill", "Bill Sikes"]

# Initialize an empty DataFrame to store sentiment analysis results
columns = ['Character', 'Positive', 'Negative', 'Neutral', 'Compound']
character_sentiments = {character: {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Compound': 0, 'Count': 0} for character in characters}

# Sentiment analysis of both authors’ text
for author, text in authors_texts.items():
    lines = text.split('\n')
    for line in lines:
        for character in characters:
            if character in line:
                sentiment_scores = analyze_sentiment_vader(line)
                # Accumulate each sentiment score
                character_sentiments[character]['Positive'] += sentiment_scores['pos']
                character_sentiments[character]['Negative'] += sentiment_scores['neg']
                character_sentiments[character]['Neutral'] += sentiment_scores['neu']
                character_sentiments[character]['Compound'] += sentiment_scores['compound']
                # Record the number of sentences to calculate the average score
                character_sentiments[character]['Count'] += 1

df_list = []

for character, scores in character_sentiments.items():
    if scores['Count'] > 0:  # Avoid dividing by zero, tricky part
        averages = {score: total/scores['Count'] for score, total in scores.items() if score != 'Count'}
        df_list.append(pd.DataFrame([averages], index=[character]))

# Merge all DataFrames
character_sentiment_averages = pd.concat(df_list)

# Print the average emotion score for each character
print(character_sentiment_averages)

"""# 3. Conclusions
#### From coursework 1 to coursework 2, we have examined different techniques in sentiment analysis, and those results are fitted even though we use different approaches from CW 1 and CW 2. We have not read all of the works of Jane Austen and Charles Dickens, but we can still understand the sentiment orientation of these results, proving that sentiment analysis in NLP is worth to research.

# 4. References
#### Python Natural Language Processing Cookbook: Over 50 recipes to understand, analyze, and generate text for implementing language processing tasks. 2021. Zhenya Antić.
#### Python-Natural-Language-Processing-Cookbook. Available: https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook
#### Python和NLTK自然语言处理[印度] 2019. Nitin Hardeniya. Available: https://github.com/packtpublishing/hands-on-natural-language-processing-with-pytorch
#### Introducing Python. Bill Lubanovic. 2019
#### Sentiment Analysis: A Definitive Guide. Available: https://monkeylearn.com/sentiment-analysis/
#### Copyright, Designs and Patents Act 1988. Available: https://www.legislation.gov.uk/ukpga/1988/48/contents
#### List Comprehension in Python. Available: https://www.youtube.com/watch?v=l8mWvDUwOt4
#### THIS is Why List Comprehension is SO Efficient! Available: https://www.youtube.com/watch?v=U88M8YbAzQk
#### Python的各种推导式（列表推导式、字典推导式、集合推导式）. Available: https://blog.csdn.net/yjk13703623757/article/details/79490476
#### Python中的三种推导式及用法. Available: https://blog.csdn.net/qq_38727995/article/details/127432353
#### When to Use a List Comprehension in Python. Available: https://realpython.com/list-comprehension-python/
#### How to Use enumerate() in Python. Available: https://www.youtube.com/watch?v=uPT-LkYSP_o
#### Python Enumerate Function - Python Quick Tips. Available: https://www.youtube.com/watch?v=-MZiQaNI0QA
#### python enumerate用法总结. Available: https://blog.csdn.net/churximi/article/details/51648388
#### python enumerate( )函数用法. Available: https://zhuanlan.zhihu.com/p/61077440
#### Radar chart (aka spider or star chart). Available: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
#### How to make a polygon radar (spider) chart in python. Available: https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
#### 視覺化：雷達圖. Available: https://orcahmlee.github.io/data-science/radar-chart/
#### Python Radar Charts (AKA - Spider Charts, Star Charts). Available: https://www.youtube.com/watch?v=JC3PKoIX0ZE
#### "Yet" Available: https://dictionary.cambridge.org/us/grammar/british-grammar/yet
#### Jane Austen was right: Her characters were known for falling ill... but historians say 19th century people WERE prone to disease, NICK MCDERMOTT, SCIENCE REPORTER, 2013, Available: https://www.dailymail.co.uk/sciencetech/article-2337371/Jane-Austen-right-Her-characters-known-falling-ill--historians-say-19th-century-people-WERE-prone-disease.html
#### An Analytical Approach for Extracting Entities and Their Emotional Tones in Narrative Scenarios, V. Ashwanth, 2022. Available: https://doi.org/10.1007/978-981-19-2130-8_16
#### An Ensemble Framework for Dynamic Character Relationship Sentiment in Fiction, Nikolaus Nova Parulian, 2022, Available: https://doi.org/10.1007/978-3-030-96957-8_35
#### Park, J., Choi, W., Jung, S-U., (2022), ‘Exploring Trends in Environmental, Social, and Governance Themes and Their Sentimental Value Over Time’, Front. Psychol., Sec. Organizational Psychology, Volume 13. Available at: https://doi.org/10.3389/fpsyg.2022.890435 (Accessed: 20 August 2023)
#### Repo P., Matschoss K., Mykkänen J. (2021), Examining outlooks on sustainability transitions through computational language analysis, Environmental Innovation and Societal Transitions, Volume 41, Pages 74-76 [online]. https://doi.org/10.1016/j.eist.2021.10.028 (Accessed: 20 August 2023)
#### Savin I., Drews S., van den Bergh J. (2021). Free associations of citizens and scientists with economic and green growth: A computational-linguistics analysis, Ecological Economics, Volume 180 (106878) [online]. Available at: https://doi.org/10.1016/j.ecolecon.2020.106878 (Accessed: 20 August 2023).
"""