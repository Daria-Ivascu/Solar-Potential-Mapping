import requests, os, zipfile
from getpass import getpass
from tqdm import tqdm

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
    "maxRecords": 100,
    "productType": "S2MSI2A" 
}

# Headers
headers = {"Authorization": f"Bearer {token}"}

# Request for the necessary products
search_response = requests.get(search_url, params=params, headers=headers)
search_response.raise_for_status()
results = search_response.json()

# Download and extraction for the products
def download_and_extract(product, headers, directory="downloaded data"):
    title = product["properties"]["title"]
    product_id = product["id"]
    extract_path = os.path.join(directory, title)  
    zip_file_path = os.path.join(directory, f"{title}.zip")

    os.makedirs(directory, exist_ok=True)

    # Checks if the .SAFE is already downloaded and downloads the data if not
    if not os.path.exists(zip_file_path):
        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            with open(zip_file_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=title, ascii=True) as pbar:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Checks if .SAFE is already downloaded and extracts the data if not
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                parts = member.split("/", 1)
                member_target = parts[1] if len(parts) > 1 else parts[0]
                if member_target:
                    target_path = os.path.join(extract_path, member_target)
                    if member.endswith("/"):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with zip_ref.open(member) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())

    print(f"Downloaded and extracted: {title}")
    return extract_path

# Downloads all the products
for product in results['features']:
    download_and_extract(product, headers)
