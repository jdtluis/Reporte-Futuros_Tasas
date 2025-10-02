import requests
import pandas as pd

# API endpoint
url = "https://apicem.matbarofex.com.ar/api/v2/closing-prices"

# Query parameters
params = {
    "product": "DLR",
    "segment": "Monedas",
    "type": "FUT",
    "excludeEmptyVol": "true",
    "from": "2025-09-02",
    "to": "2025-09-29",
    "page": "1",
    "pageSize": "50",
    "sortDir": "ASC",
    "market": "ROFX"
}

# Send GET request
response = requests.get(url, params=params)

# Check if request was successful
if response.status_code == 200:
    data = response.json()
    
    # Often APIs return data under a key like "content" or "results"
    # If so, replace 'data' below with data["content"]
    try:
        df = pd.DataFrame(data['data'])
    except ValueError:
        df = pd.json_normalize(data['data'])
    
    print(df.head())
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)



# BYMA DATA API
url_simul = 'https://new2.bymadata.com.ar/vanoms-be-core/rest/api/byma/data/quantex/trades-volumes'
payload = '{"layout_oid":4375,"page_number":1,"Content-Type":"application/json"}'
