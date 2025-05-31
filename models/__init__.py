# This file makes Python treat the 'models' directory as a package.
# It can also be used to make sub-modules more easily accessible.

# Example:
# from .vision import SimpleCNN # If SimpleCNN was directly in vision/__init__.py
# from .nlp import load_bert_classifier # If load_bert_classifier was in nlp/__init__.py

# Or, if you want to expose everything from the sub-package's __init__.py:
# from .vision import *
# from .nlp import *

# For now, just ensuring the submodules are seen as part of the models package.
# Specific imports will be handled by the train.py script directly from models.vision or models.nlp
pass
