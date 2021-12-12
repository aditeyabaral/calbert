import time
import json
import requests
import pprint
import pandas as pd
from flask import Flask, request, render_template,jsonify

app = Flask(__name__,template_folder="templates")

API_URLS = [
            "https://api-inference.huggingface.co/models/aditeyabaral/finetuned-sail2017-xlm-roberta-base",
            "https://api-inference.huggingface.co/models/aditeyabaral/finetuned-iitp_pdt_review-xlm-roberta-base"
]
headers = {"Authorization": "Bearer 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"}

def get_sentiment(sentence):
  output = list()
  payload = {"inputs": sentence}
  for API_URL in API_URLS:
    response = requests.post(API_URL, headers=headers, json=payload)
    response = response.json()
    while "error" in response and "estimated_time" in response:
      time_to_wait = response["estimated_time"]
      print(f"Waiting for {time_to_wait} s")
      time.sleep(time_to_wait + 2)
      response = requests.post(API_URL, headers=headers, json=payload)
      response = response.json()
    output.append(response)
  return output

def process_output(output):
  result = dict()
  model_names = ["SAIL2017", "IIT-P Product Reviews"]
  label_map = {'0': "NEGATIVE", '1': "NEUTRAL", '2': "POSITIVE"}
  for model_output, model_name in zip(output, model_names):
    result[model_name] = dict()
    predictions = model_output[0]
    predictions.sort(key=lambda x: x["score"], reverse=True)
    result[model_name]["prediction"] = dict()
    for prediction in predictions:
      result[model_name]["prediction"][label_map[prediction["label"][-1]]] = prediction["score"]
    max_prediction_score = predictions[0]["score"]
    max_prediction_label = predictions[0]["label"]
    label_name = label_map[max_prediction_label[-1]]
    result[model_name]["max_label"] = label_name
    result[model_name]["max_probability"] = max_prediction_score
  return result


pp = pprint.PrettyPrinter(indent=4)
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/inference',methods=['POST','GET'])
def classify_type():
    try:
        input_str = request.form.get('inputstr') 
        print(input_str)
        output = get_sentiment(input_str)
        output = process_output(output)
        print(output)
        return render_template('output.html', output = output)

    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        