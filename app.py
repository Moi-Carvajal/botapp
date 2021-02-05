
#pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#curl -d "{\"oracion\":[definición de ahorro]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predict
#curl -d '{oracion:"definición de ahorro"}' -H 'Content-Type: application/json' -X POST http://127.0.0.1:5000/predict
import transformers
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from flask import Flask, jsonify, request
import pandas as pd
import torch

# import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss   
# import numpy as np

from arquitectura import BertForSequenceClassification
tokenizer = torch.load('tokenizer')
t_model = torch.load('Modelo_final',map_location=torch.device('cpu'))
db=pd.read_excel("db.xlsx")
app= Flask(__name__)

@app.route("/")
def home():
    return 'Ready!'

@app.route("/predecir", methods=["POST"])
def predecir(tokenizer=tokenizer,t_model=t_model,db=db):
    json_doc=request.get_json(force=True)
    # if not json_doc:
    #   return jsonify(error="request body cannot be empty"), 400
    sentence=json_doc.get('oracion')
    #---------------------------------------------------------------------------
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encoding = tokenizer(str(sentence.lower()), return_tensors='pt', padding=True, truncation=True)
    texto_ids = encoding['input_ids']
    t_model.eval()
    predict = t_model(texto_ids)
    Etiqueta= torch.argmax(predict[0]).item()
    #---------------------------------------------------------------------------
    db=pd.read_excel("db.xlsx")
    value=db['respuesta'][Etiqueta]
    if Etiqueta==0:
      v_f=[]
      for i in range(1,len(value)):
        v_f.append(db['respuesta'][i])
      value=v_f
    #---------------------------------------------------------------------------
    resp={"Respuesta":value}
    return jsonify(resp)

if __name__ == '__main__':
    app.run(debug=True)