# -*- coding: utf-8 -*-
import re
import time
from string import punctuation
from nltk.tokenize import word_tokenize
# from translate import Translator
from selenium import webdriver
import logging
from selenium.webdriver.remote.remote_connection import LOGGER
from selenium.webdriver.chrome.options import Options

# translator = Translator(from_lang="hindi", to_lang="english")

LOGGER.setLevel(logging.FATAL)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--log-level=3")


def display(source, translation, transliteration, cleaned_transliteration):
    print(
        f"\nSOURCE: {source}\n\nTRANSLITERATION: {transliteration}\n\nCLEANED TRANSLITERATION: {cleaned_transliteration}\n\nTRANSLATION: {translation}")


def join_punctuation(seq, characters=punctuation):
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current


def cleanTransliteration(source, transliteration):
    # handle punctuation to this list?
    pattern = re.compile(r"[\u0900-\u097F]")
    source_words = word_tokenize(source)
    transliteration_words = word_tokenize(transliteration)
    cleaned_transliteration = [t_word if re.findall(
        pattern, source_words[position]) else source_words[position] for position, t_word in enumerate(transliteration_words)]
    cleaned_transliteration = ' '.join(
        join_punctuation(cleaned_transliteration)).strip()
    # print(cleaned_transliteration)
    return cleaned_transliteration


def getConvertedText(source):
    url = f"https://translate.google.com/?hl=en&sl=hi&tl=en&text={source}&op=translate"

    chrome = webdriver.Chrome(
        executable_path=r"/usr/bin/chromedriver", options=chrome_options)
    chrome.get(url)
    time.sleep(1)

    sourcebox = chrome.find_element_by_xpath(r'//*[@aria-label="Source text"]')
    # translation = translator.translate(source)    # has a limited number of translations per day - around 10
    translation = chrome.find_element_by_class_name(r'J0lOec').text
    transliteration = chrome.find_element_by_class_name(r'kO6q6e').text
    try:
        cleaned_transliteration = cleanTransliteration(source, transliteration)
    except:
        cleaned_transliteration = transliteration

    # display(source, translation, transliteration, cleaned_transliteration)

    return {
        "source": source,
        "translation": translation,
        "transliteration": transliteration,
        "cleaned-transliteration": cleaned_transliteration
    }

# source = '''रजिस्ट्रार ऑफ कंपनीज के पास दाखिल शाओमी की फाइलिंग्स के मुताबिक, बीजिंग की इस कंपनी ने कहा कि वह 'ट्रांसपोर्ट, कन्वेएंस के लिए सभी तरह के वीइकल्स बेच सकती है, चाहे वे इलेक्ट्रिसिटी बेस्ड हों या मैकेनिकल पावर बेस्ड।
# ' उसने इसके अलावा 'ट्रांसपोर्ट इक्विपमेंट, आटो कंपोनेंट्स और स्पेयर पार्ट्स बेचने' की संभावना भी जताई है।'''
# result = getConvertedText(source)
