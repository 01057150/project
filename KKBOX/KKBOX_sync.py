import pandas as pd
import requests
import urllib.parse
import os
import logging
import time

# Define the access token
access_token = 't-mYccVzLtTTJwN0TMtprg=='
RATE_LIMIT = 0.01  # 100 milliseconds
MAX_RETRIES = 5
API_COOLDOWN_PERIOD = 600  # 10 minutes cooldown period in seconds

logging.basicConfig(filename='kkbox_api_errors.log', level=logging.ERROR)

def search_kkbox_api(query, retries=1, delay=2):
    url = f"https://api.kkbox.com/v1.1/search?q={urllib.parse.quote(query)}&territory=TW&offset=0&limit=5"
    headers = {
        'accept': "application/json",
        'authorization': f"Bearer {access_token}"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                if 'application/json' in response.headers.get('Content-Type', ''):
                    return response.json()
                else:
                    logging.error(f"Unexpected content type: {response.headers.get('Content-Type')} for query: {query}")
                    logging.info(f"Message: {response.text}")
                    return {}
            else:
                logging.error(f"Unexpected status code: {response.status_code} for query: {query}")
                logging.info(f"Message: {response.text}")

                if response.status_code == 500 and attempt < retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return {}
        except requests.RequestException as e:
            logging.error(f"Client error: {e} for query: {query}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return {}
        except Exception as e:
            logging.error(f"Unexpected error: {e} for query: {query}")
            return {}

def process_row(index, row):
    artist_name = row['artist_name']
    song_name = row['song_name']
    isrc = row['isrc']
    query = f"{song_name} {artist_name}"
    
    try:
        result = search_kkbox_api(query)
        if result == 'rate_limit_exceeded':
            return 'rate_limit_exceeded'

        track_data_list = result.get('tracks', {}).get('data', [])
        
        if not track_data_list:
            return index, None, None, None
        
        # Find the correct track based on ISRC
        correct_track = None
        for track in track_data_list:
            if track.get('isrc', '') == isrc:
                correct_track = track
                break

        # If no correct track is found and isrc is 'Unknown', use the first result
        if not correct_track and isrc == 'Unknown':
            correct_track = track_data_list[0]

        if not correct_track:
            return index, None, None, None
        
        track_id = correct_track.get('id', '')
        track_isrc = correct_track.get('isrc', '')
        album_images = correct_track.get('album', {}).get('images', [])
        
        album_image_500x500 = ''
        for image in album_images:
            if image['height'] == 500 and image['width'] == 500:
                album_image_500x500 = image['url']
                break
        
        return index, track_id, track_isrc, album_image_500x500
    
    except Exception as e:
        logging.error(f"Error processing {query}: {e}")
        return index, None, None, None

def process_file(file_index):
    extend_file = f'D:/song_data_extend_{file_index + 1:03d}.csv'
    split_file = f'D:/split_song/output_file_{file_index + 1:03d}.csv'

    if os.path.exists(extend_file):
        print(f'Opening {extend_file}')
        df = pd.read_csv(extend_file, encoding='utf-8')
    else:
        print(f'Opening {split_file}')
        df = pd.read_csv(split_file, encoding='utf-8')
        df['track_id'] = ''
        df['track_isrc'] = ''
        df['album_image'] = ''
    
    df_to_process = df[df['track_isrc'].isnull() | (df['track_isrc'] == '')]
    
    # Print number of rows to be processed
    print(f"Number of rows to process: {len(df_to_process)}")
    
    correct_count = 0
    for index, row in df_to_process.iterrows():
        result = process_row(index, row)
        if result == 'rate_limit_exceeded':
            print("Rate limit exceeded. Waiting for 10 minutes before retrying...")
            time.sleep(API_COOLDOWN_PERIOD)
            return False  # Indicate the need to retry
        
        index, track_id, track_isrc, album_image = result
        if track_id is not None:
            df.at[index, 'track_id'] = track_id
            df.at[index, 'track_isrc'] = track_isrc
            df.at[index, 'album_image'] = album_image
            correct_count += 1
            
    print(correct_count)
    
    df.to_csv(f'D:/song_data_extend_{file_index + 1:03d}.csv', index=False)
    return True

def main():
    for j in range(5):
        for i in range(25, 36):
            for attempt in range(MAX_RETRIES):
                success = process_file(i)
                if success:
                    break
                else:
                    print(f"Retrying file {i + 1:03d} (Attempt {attempt + 1}/{MAX_RETRIES})")

if __name__ == "__main__":
    main()
