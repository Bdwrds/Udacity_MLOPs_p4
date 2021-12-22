import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:"
PORT = 8000
QUERY = "?csv_path=/home/dora/github/Udacity_MLOPs_p4/testdata/testdata.csv"

#Call each API endpoint and store the responses
response1 = requests.post(URL + str(PORT) + "/prediction" + QUERY).content
response2 = requests.get(URL + str(PORT) + "/scoring" + QUERY).content
response3 = requests.get(URL + str(PORT) + "/summarystats" + QUERY).content
response4 = requests.get(URL + str(PORT) + "/diagnostics" + QUERY).content

#combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4,
             }

#write the responses to your workspace
print(responses)


