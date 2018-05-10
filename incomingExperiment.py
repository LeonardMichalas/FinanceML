import urllib2
import json

data = {
        "Inputs": {
                "input1":
                [
                    {
                            'age': "44",   
                            'education': "Doctorate",   
                            'marital-status': "Married-civ-spouse",   
                            'relationship': "Own-child",   
                            'race': "white",   
                            'sex': "female",   
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/8340e4cb4beb425cbd2789f0c5f90dae/services/df312316161b413e93e1c6ddd0c6a06a/execute?api-version=2.0&format=swagger'
api_key = 'gpPh0oHbekLlLfcL40GaPE/w2hCNLjnxyu7ryPvBj8Z2OAkfj1195cWQtzp8DEo6FlS3+Lb1qqiuHu4Tn6ZnHA==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers)

try:
    response = urllib2.urlopen(req)

    result = response.read()
    print(result)
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read())) 

