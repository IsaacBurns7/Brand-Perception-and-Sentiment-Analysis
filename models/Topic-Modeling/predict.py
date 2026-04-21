#predicts using model in cache/model_name on dataframe (maybe streaming??)
import importlib
nltk_module = importlib.import_module("nltk")
nltk_module.download('stopwords', quiet=True)
nltk_module.download('punkt', quiet=True)