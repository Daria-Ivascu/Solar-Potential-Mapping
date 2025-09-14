import requests
from getpass import getpass

username = input("CDSE Username: ")
password = getpass("CDSE Password: ")

token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
data = {
    "client_id": "cdse-public",
    "grant_type": "password",
    "username": username,
    "password": password,
}

# Gets the token
response = requests.post(token_url, data=data, verify=True, allow_redirects=False)
response.raise_for_status()
token = response.json()['access_token']

# Searching URL
search_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/SENTINEL-2/search.json"

# Searching params
params = {
    "startDate": "2023-01-01T00:00:00Z",
    "completionDate": "2023-12-31T23:59:59Z",
    "geometry": "POLYGON((26 44, 27 44, 27 45, 26 45, 26 44))",
    "maxRecords": 1,
    "productType": "S2MSI2A" 
}

# Headers
headers = {"Authorization": f"Bearer {token}"}

# Request for the necessary products
search_response = requests.get(search_url, params=params, headers=headers)
search_response.raise_for_status()
results = search_response.json()

# Extracts produt's ID
product = results['features'][0]
title = product['properties']['title']
product_id = product['id']

# Downloading link for the product
download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

print("Found product:", title)
print("Product's download link:", download_url)

# Downloads the data
r = requests.get(download_url, headers=headers, stream=True)
r.raise_for_status()

with open(f"{title}.zip", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Check download:", title)
