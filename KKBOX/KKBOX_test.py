import http.client
import urllib.parse
import requests
import json

# Define the access token
access_token = 't-mYccVzLtTTJwN0TMtprg=='

# Function to search KKBOX API

def search_kkbox_api(query, territory="TW", offset=0, limit=5):
    url = f"https://api.kkbox.com/v1.1/search?q={urllib.parse.quote(query)}&territory={territory}&offset={offset}&limit={limit}"
    headers = {
        'accept': "application/json",
        'authorization': f"Bearer {access_token}"  # 確保此處填入有效的 access_token
    }
    
    response = requests.get(url, headers=headers)
    data = response.content

    # 打印響應內容以檢查
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {data}")

    try:
        # 僅在響應內容類型為 JSON 且數據非空時解析
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            json_data = json.loads(data.decode("utf-8"))
            return json_data
        else:
            print(f"Unexpected content type or empty response for query: {query}")
            return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e} for query: {query}")
        print(f"Response content: {data.decode('utf-8') if data else 'No content'}")
        return {}
    except Exception as e:
        print(f"Error: {e} for query: {query}")
        return {}

# Example usage
query = "The Dream"
#type = "track,album,artist,playlist"
territory = "SG"
offset = 0
limit = 5

response = search_kkbox_api(query, territory, offset, limit)

print(json.dumps(response, indent=4, ensure_ascii=False))
