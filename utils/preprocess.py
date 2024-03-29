
import string
import re
from nltk.corpus import stopwords

class MakeString:
    def process(self, text):
        return str(text)
    

class ReplaceBy:
    def __init__(self, replace_by):
        #replace_by is a tuple contains pairs of replace and by characters.
        self.replace_by = replace_by
    def process(self, text):
        for replace, by in self.replace_by:
            text = text.replace(replace, by)
        return text

class LowerText:
    def process(self, text):
        return text.lower()


class ReduceTextLength:
    def __init__(self, limited_text_length):
        self.limited_text_length = limited_text_length
    def process(self, text):
        return text[:self.limited_text_length]


class VectorizeText:
    def __init__(self):
        pass
    def process(self, text):
        return text.split()


class FilterPunctuation:
    def __init__(self):
        print("Punctuation Filter created...")
    def process(self, words_vector):
        reg_exp_filter_rule=re.compile("[%s]"%re.escape(string.punctuation))
        words_vector=[reg_exp_filter_rule.sub("", word) for word in words_vector]
        return words_vector

class FilterNonalpha:
    def __init__(self):
        print("Nonalpha Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if word.isalpha()]
        return words_vector

class FilterStopWord:
    def __init__(self, language):
        self.language=language
        print("Stopwords Filter created...")
    def process(self, words_vector):
        stop_words=set(stopwords.words(self.language))
        words_vector=[word for word in words_vector if not word in stop_words]
        return words_vector


class FilterShortWord:
    def __init__(self, min_length):
        self.min_length=min_length
        print("Short Words Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if len(word)>=self.min_length]
        return words_vector 

class TextProcessor:
    def __init__(self, processor_list):
        self.processor_list = processor_list
    def process(self, text):
        for processor in self.processor_list:
            text = processor.process(text)
        return text

class JoinWithSpace:
    def __init__(self):
        pass
    def process(self, words_vector):
        return " ".join(words_vector)
    
makeString = MakeString()

replace_by = [("."," "), ("?"," "), (","," "), ("!"," "),(":"," "),(";"," "),
              ("'"," "),('"'," ")]
replaceBy =ReplaceBy(replace_by=replace_by)

lowerText = LowerText()

FACTOR=8
mean_review_length = 3000
reduceTextLength = ReduceTextLength(limited_text_length=mean_review_length*FACTOR)

vectorizeText = VectorizeText()
filterPunctuation = FilterPunctuation()
filterNonalpha = FilterNonalpha()
filterStopWord = FilterStopWord(language = "turkish")

min_length = 2
filterShortWord = FilterShortWord(min_length=min_length)
joinWithSpace = JoinWithSpace()
processor_list_1 = [makeString,
                      replaceBy,
                      lowerText,
                      reduceTextLength,
                      vectorizeText,
                      filterPunctuation,
                      filterNonalpha,
                      filterStopWord,
                      filterShortWord,
                      joinWithSpace]

textProcessor1 = TextProcessor(processor_list=processor_list_1)


if __name__ == "__main__":
    text = "Bana bir şeyler Söyle'i bu şu kelimeleri ve veya cümleleri"
    text_processed = textProcessor1.process(text)
    print("text:", text)
    print("processed text:", text_processed)