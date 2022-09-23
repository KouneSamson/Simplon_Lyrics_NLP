import requests
import json

def call_service(url,data):
    request_body = json.dumps(data)
    headers = {"content-type": "application/json"}
    try:
        r = requests.post(url, data=request_body, headers=headers)
    except requests.Timeout:
        print(f"CS:T1 / {url} / {request_body[:250]}")
        raise Exception
    except requests.ConnectionError as errce:
        print(f"CS:T2 / {errce} / {url} / {request_body[:250]}")
        raise Exception
    return json.loads(r.content.decode('utf-8'))

phrase = "Got a head full of noise About a hundred different things I'm tryna avoid I got a mind in the gutter"
data = dict(phrase=phrase.lower())
result = call_service("http://localhost:5000/prediction/LSTM", data)
print(f"Input  ::\n{result['seq_in']}")
print(f"\nOutput ::\n{result['result']}")