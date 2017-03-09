from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearm.pipeline import Pipeline

naive_count_clf = Pipeline([('vect',CountVectorizer()),('clf', MultinomialNB(alpha=1, fit_prior='false'))])
