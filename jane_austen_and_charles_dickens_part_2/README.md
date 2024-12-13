# Writing Style Differences between Jane Austen and Charles Dickens Part 2:  

## Introduction

This project builds upon earlier research to further examine the stylistic and emotional characteristics found in the works of two renowned British authors, **Jane Austen** and **Charles Dickens**. In Part 1 of our study, we discovered that Austen’s writing tends to exhibit a more positive overall sentiment, while Dickens’s works display greater emotional fluctuations and a higher proportion of neutral sentiment. In Part 2, we aim to deepen this analysis by exploring additional dimensions, including syntactic structure, part-of-speech distribution (adjectives, adverbs, and verbs), and sentiment analysis with compound and subjectivity indices. Advanced techniques such as TF-IDF, Named Entity Recognition (NER), and semantic search using BERT will also be applied.

### **Key Objectives:**
- Analyze the syntactic structures and part-of-speech distributions of the two authors.
- Conduct sentiment analysis enriched by compound and subjectivity indices.
- Explore the application of TF-IDF to identify term importance.
- Perform Named Entity Recognition (NER) to analyze character-based sentiment.
- Utilize semantic search with BERT to uncover deeper semantic relationships.

### **Significance:**
By integrating traditional and advanced Natural Language Processing (NLP) methods, this project aims to provide:
- A more comprehensive understanding of the stylistic and emotional tendencies in Austen’s and Dickens’s works.
- Insights into how vocabulary, syntax, and character usage influence the emotional dynamics of their texts.
- A foundation for developing automated writing style simulations, enhanced text sentiment analysis, and literary research tools.

## Libraries Used

### 1. Core Libraries
- **os**: Handles file paths and directory management.
- **collections.Counter**: Counts elements (e.g., words, POS tags, syntactic structures).
- **random**: Used for random sampling when reducing dataset size.

### 2. NLP Libraries
- **SpaCy**:
  - `nlp` for text parsing and syntactic analysis.
  - `displacy` for visualizing dependency trees.
- **NLTK (Natural Language Toolkit)**:
  - `word_tokenize`: Splits text into individual tokens.
  - `pos_tag`: Tags words with grammatical categories.
  - `ne_chunk`: Extracts named entities for NER tasks.
  - `SentimentIntensityAnalyzer`: Performs polarity scoring for sentiment analysis.
  - `stopwords`: Filters out common stopwords.
- **TextBlob**:
  - Provides polarity and subjectivity scores for a more nuanced sentiment analysis.

### 3. Machine Learning Libraries
- **scikit-learn**:
  - `TfidfVectorizer`: Calculates TF-IDF scores to identify important words and features in the text.

### 4. Data Manipulation and Analysis
- **pandas**:
  - Used to create, manipulate, and analyze structured data (e.g., sentiment scores, TF-IDF matrices).
- **NumPy**:
  - Supports numerical operations and data preparation for visualization.

### 5. Data Visualization
- **matplotlib**:
  - Produces line charts, bar plots, radar charts, and other visualizations.
- **WordCloud**:
  - Generates visual representations of word frequencies, segmented by POS and other criteria.

### 6. Google Colab Integration
- **Google Colab Drive API**:
  - Facilitates loading and saving of text files directly from Google Drive.

### 7. Supporting Utilities
- **re (Regular Expressions)**:
  - Cleans text by removing punctuation, special characters, and unnecessary whitespace.
- **PorterStemmer**:
  - Stems words to their root forms, aiding in TF-IDF analysis and ensuring data consistency.

### 8. External Resources
- **SpaCy Model**:
  - `en_core_web_sm`: A pre-trained English model for POS tagging and dependency parsing.
- **NLTK Resources**:
  - `vader_lexicon`: Provides lexicon-based sentiment scoring.
  - `averaged_perceptron_tagger`, `stopwords`, `words`: Pre-trained models and datasets for tokenization, POS tagging, and stopword filtering.

---
