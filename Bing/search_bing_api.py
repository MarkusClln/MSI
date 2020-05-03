import requests
import os
import shutil


subscription_key = "" #<--- ur bing api key here
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
search_term = "Tom Cruise"
search_times = 15

headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

params = {"q": search_term, "license": "public", "imageType": "photo"}

response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:search_times]]

try:
    os.mkdir('images/'+search_term)
except FileExistsError:
    print("Folder exists")
count = 0;
for url in thumbnail_urls:
    # try to download the image
    try:
        print("[INFO] fetching: {}".format(url))
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200:
            with open('images/'+search_term+"/"+str(count)+".png", 'wb') as f:
                count +=1
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    # catch any errors that would not unable us to download the image
    except Exception as e:
        print(e)
        continue