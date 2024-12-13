# Writing Style Differences between Jane Austen and Charles Dickens Part 1:  

## Introduction

This project investigates the emotional characteristics present in the writings of two renowned British authors, **Jane Austen** and **Charles Dickens**, through text mining and sentiment analysis. By obtaining paragraph-level datasets from Kaggle, filtering them, and reducing the data size, we conducted polarity (positive/negative) and subjectivity analyses. Various visualization tools, such as radar charts, box plots, and word clouds, were employed to present the findings.

**Key Objectives:**
- Compare the emotional tendencies in the writing styles of Jane Austen and Charles Dickens.
- Verify common literary perspectives:  
  - Austen’s works are often characterized by humor, generally positive sentiments, and balanced emotional fluctuations.
  - Dickens’s works frequently show greater emotional variation, focusing on themes like social injustice, poverty, and the complexity of human nature.

**Significance:**
The results of this study provide insights that can be applied to:
- Developing automated writing style simulation tools.
- Conducting more nuanced literary text analyses.
- Gaining a deeper understanding of historical contexts, authorial backgrounds, and potential modern reader receptions.

## Libraries Used

1. **os**  
   - Handles file path and directory management for reading and writing files.
   
2. **random**  
   - Used to randomly sample lines from text files when reducing dataset size.
   
3. **pandas (pd)**  
   - Creates and manipulates data frames for organizing text and sentiment data.
   
4. **nltk**  
   - `nltk.corpus.stopwords`: Filters out stopwords from text data.  
   - `nltk.sentiment.vader.SentimentIntensityAnalyzer`: Performs sentiment analysis on the text.  
   - `nltk.tokenize.word_tokenize`: Tokenizes text into individual words.
   
5. **re**  
   - Provides regular expressions for cleaning and preprocessing text data by removing unwanted characters.
   
6. **matplotlib.pyplot**  
   - Plots graphs, including sentiment trend lines and box plots.
   
7. **seaborn**  
   - Enhances data visualization with more appealing and informative plots (especially for box plots).
   
8. **wordcloud.WordCloud**  
   - Generates word clouds to visually represent word frequency and prominence.

---
