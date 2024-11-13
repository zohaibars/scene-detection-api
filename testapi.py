import requests
# Replace with the URL where your FastAPI server is running
url = "http://"+"192.168.18.81"+":2008"
# print(url)
# Replace with your API key
api_key = "apikey1"
def SceneRecognition(file_path,ID=None):
    try:
        # print('API IS LIVE')
        with open(file_path, 'rb') as file_path_:
            files = {'file': file_path_}
            headers={
                'api_key': api_key
            }
            params = {'ID': ID}
            response = requests.post(f"{url}/Scene_Recognition/", files=files, headers=headers,params=params)
        if response.status_code == 200:
            result = response.json()
            print(result)
        else:
            print(response.text)
    except Exception as e:
        print("Error:", str(e))
if __name__ == "__main__":
    SceneRecognition(r"TESTData\image\aa.png",ID=5)

