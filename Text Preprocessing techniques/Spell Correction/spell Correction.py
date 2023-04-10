# pip install pyspellchecker textblob

from textblob import TextBlob
from spellchecker import SpellChecker

# Create a TextBlob object with the text to correct
text = "I havv a goood speling!"

# Create a new TextBlob object with corrected spelling
corrected_text = TextBlob(text).correct()

# Print the corrected text
print(corrected_text)

print("---------------------------")


# Create a SpellChecker object
spell = SpellChecker()

# Create a list of words with misspelled words
misspelled_words = ["recieve", "acknowlegement", "seperate"]

# Iterate over the list and correct the misspelled words
for word in misspelled_words:
    corrected_word = spell.correction(word)
    print(f"Original word: {word}, Corrected word: {corrected_word}")