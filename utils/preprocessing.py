import re
import json
import pandas as pd


def cleanText(text):
    text = text.lower().strip()
    text = re.sub("\d+(\.\d+|(,\d+)+\d+){0,1}", "<NUM>", text)
    text = re.sub(" +", " ", text)
    text = re.sub("\n+", " ", text)
    return text


def cleanData(data):
    new_data = list()
    for dic in data:
        tmp_dict = dict()
        for key in dic:
            tmp_dict[key] = cleanText(dic[key])
        new_data.append(tmp_dict)
    return new_data


def createMergedDataFrame(df_list):
    return pd.concat(df_list)


def createDataFrame(path, clean=False, verbose=True):
    with open(path) as f:
        data = json.load(f)

    if clean:
        data = cleanData(data)

    cleaned_transliteration = list()
    source = list()
    translation = list()
    transliteration = list()

    pattern = re.compile(r"[.]")

    for i in data:
        t1 = [t for t in re.split(
            pattern, i["cleaned-transliteration"])if t not in ['', ' ', '"', "'"]]
        t2 = [t for t in re.split(pattern, i["source"]) if t not in [
            '', ' ', '"', "'"]]
        t3 = [t for t in re.split(pattern, i["translation"]) if t not in [
            '', ' ', '"', "'"]]
        t4 = [t for t in re.split(pattern, i["transliteration"]) if t not in [
            '', ' ', '"', "'"]]
        if len(t4) != len(t3):
            # handle this
            if verbose:
                print(i["source"])
                print(len(t1), len(t2), len(t3), len(t4))
                print(f"{t1}\n{t2}\n{t3}\n{t4}")
        else:
            cleaned_transliteration.extend(t1)
            source.extend(t2)
            translation.extend(t3)
            transliteration.extend(t4)

    df = pd.DataFrame()
    #df["source"] = source
    df["translation"] = translation
    df["transliteration"] = transliteration
    #df["cleaned_transliteration"] = cleaned_transliteration
    return df
