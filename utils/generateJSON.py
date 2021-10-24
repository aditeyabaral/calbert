import time
import convert
import sys
import json
import traceback
from nltk.tokenize import sent_tokenize

files = ["../data/sources/" + f for f in sys.argv[1:]]
total_files = len(files)

failures = 0
total_converts = 0


def convertData1To5(fname):
    f = open(fname, "r", encoding="utf-8")
    text = f.read().strip().split("\n")
    f.close()
    output = []
    total = len(text)
    for ctr, line in enumerate(text):
        print(f"Processed {ctr+1} lines out of {total} in file {fname}.txt")
        sents = sent_tokenize(line)
        for sent in sents:
            trans = convert.getConvertedText(sent)
            if trans["source"].lower() == trans["translation"].lower():
                print("DID NOT TRANSLATE")
            else:
                output.append(trans)
    fout = open(fname[:-4]+".json", "w", encoding="utf-8")
    fout.write(json.dumps(output))


for ctr1, file in enumerate(files):
    print(f"Processing {file}")

    if "data1" in file or "data2" in file or "data3" in file or "data4" in file or "data5" in file:
        convertData1To5(file)
        continue

    converted = []
    with open(str(file), encoding="utf-8") as f:
        content = f.read().strip()
        lines = content.split('।')
        total_lines = len(lines)
        for ctr2, source_text in enumerate(lines):
            try:
                result = convert.getConvertedText(source_text)
                print(f"Converted {ctr2+1} lines out of total {total_lines} Lines")
            except Exception:
                print("FAILED")
                print(f"Failed at line {ctr2+1} in file {file}")
                traceback.print_exc()
                failures+=1
                continue
            print(result)
            total_converts+=1
            converted.append(result)
            time.sleep(2)
            
    output = json.dumps(converted)
    fout = open(file.split(".")[0]+".json", "w", encoding="utf-8")
    fout.write(output)
    fout.close()
    time.sleep(5)

print(f"{failures} FAILED CONVERTS out of {total_converts} CONVERSIONS")
