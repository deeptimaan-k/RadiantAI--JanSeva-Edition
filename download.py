import ssl
import nltk

# Disable SSL certificate verification temporarily
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt_tab')
