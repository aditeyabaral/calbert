import re
import time
import logging
from string import punctuation
from selenium import webdriver
import urllib.parse as urlparse
from urllib.parse import urlencode
from nltk.tokenize import word_tokenize
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.remote_connection import LOGGER

LOGGER.setLevel(logging.FATAL)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--log-level=3")


def getURL(base_language_code, target_language_code, source):
    url = f"https://translate.google.com/"
    params = {'hl': base_language_code, 'sl': target_language_code,
              'text': source, 'tl': 'en', 'op': 'translate'}
    url_parts = list(urlparse.urlparse(url))
    query = dict(urlparse.parse_qsl(url_parts[4]))
    query.update(params)
    url_parts[4] = urlencode(query)
    url = str(urlparse.urlunparse(url_parts))
    return url


def joinPunctuation(seq, characters=punctuation):
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
    pattern = re.compile(r"[\u0900-\u097F]")
    source_words = word_tokenize(source)
    transliteration_words = word_tokenize(transliteration)
    cleaned_transliteration = [t_word if re.findall(
        pattern, source_words[position]) else source_words[position] for position, t_word in enumerate(transliteration_words)]
    cleaned_transliteration = ' '.join(
        joinPunctuation(cleaned_transliteration)).strip()
    return cleaned_transliteration


def getConvertedText(source, base_language_code, target_language_code):
    url = getURL(base_language_code, target_language_code, source)
    chrome = webdriver.Chrome(
        executable_path=r"chromedriver.exe", options=chrome_options)
    chrome.get(url)
    time.sleep(1)

    translation = chrome.find_element_by_class_name(r'J0lOec').text
    transliteration = chrome.find_element_by_class_name(r'kO6q6e').text
    try:
        cleaned_transliteration = cleanTransliteration(source, transliteration)
    except:
        cleaned_transliteration = transliteration

    return {
        "source": source,
        "translation": translation,
        "transliteration": transliteration,
        "cleaned-transliteration": cleaned_transliteration
    }
