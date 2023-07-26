# Lexical-Substitution
Project for Natural Language Processing Course at Columbia University's School of Engineering and Applied Science, Nov 2022

Introduction:
In this assignment I worked on a lexical substitution task, using, WordNet, pre-trained Word2Vec embeddings, and BERT. This task was first proposed as a shared task at SemEval 2007 Task 10 (http://nlp.cs.swarthmore.edu/semeval/tasks/task10/description.php). In this task, the goal is to find lexical substitutes for individual target words in context. For example, given the following sentence:

"Anyway , my pants are getting tighter every day ." 

the goal is to propose an alternative word for tight, such that the meaning of the sentence is preserved. Such a substitute could be constricting, small or uncomfortable.

In the sentence

"If your money is tight don't cut corners ." 

the substitute small would not fit, and instead possible substitutes include scarce, sparse, limited, and constricted. I implemented a number of basic approaches to this problem and compared their performance.

To achieve this, I used the BERT implementation by Huggingface, or more specifically their slightly more compact model DistilBERT (https://medium.com/huggingface/distilbert-8cf3380435b5).

Packages Used:
- Natural Language Toolkit (https://www.nltk.org/) = standard way to access WordNet in Python, contains a number of useful resources, such as POS taggers and parsers, as well as access to several text corpora and other data sets.
- WordNet interface = large lexical database of English
- Gensim = vector space modeling package for Python
- Pre-trained Word2Vec embeddings = trained using a modified skip-gram architecture on 100B words of Google News text, with a context window of +-5; have 300 dimensions.
