import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import isodate
from dateutil import parser
import requests
import time
import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords','./data_science')
nltk.download('punkt', './data_science')

load_dotenv()
API_KEY = os.getenv('API_KEY')
#print(API_KEY)

api_service_name = "youtube"
api_version = "v3"

youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=API_KEY)

def get_channel_stats(CHANNEL_ID):
    """
    Get top level stats as text from channel with given ID
    Params:

    CHANNEL_ID: the input channel ID

    Returns:
    Dataframe with channel stats.

    """
    all_data = []
    request = youtube.channels().list(
        part = "snippet,statistics,contentDetails",
        id = CHANNEL_ID
    )
    response = request.execute()
    for item in response['items']:
        data = {
            'video_title' : item['snippet']['title'],
            'video_desc' : item['snippet']['description'],
            'published_at' : item['snippet']['publishedAt'],
            'view_count' : item['statistics']['viewCount'],
            'subscriber_count' : item['statistics']['subscriberCount'],
            'video_count' : item['statistics']['videoCount']
        }
        all_data.append(data)

    return pd.DataFrame(all_data)

def get_video_ids(playlist_id):

    video_ids = []

    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults = 50
    )
    response = request.execute()

    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    while next_page_token is not None:
        request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId = playlist_id,
                    maxResults = 50,
                    pageToken = next_page_token)
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')

    return video_ids

def get_video_details(video_ids):
    """
    Get top level details as text from all videos with given list of video IDs.
    Params:

    video_ids: list of video IDs

    Returns:
    Dataframe with video details  associated with video_ids.

    """
    all_video_info = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'commentCount'],
                             'contentDetails': ['duration', 'caption']
                            }
            video_info = {}
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)

    return pd.DataFrame(all_video_info)

def get_comments_in_videos(video_ids):
    """
    Get top level comments as text from all videos with given IDs (only the first 10 comments due to quote limit of Youtube API)
    Params:

    youtube: the build object from googleapiclient.discovery
    video_ids: list of video IDs

    Returns:
    Dataframe with video IDs and associated top level comment in text.

    """
    all_comments = []

    for video_id in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            )
            response = request.execute()

            comments_in_video = [comment['snippet']['topLevelComment']['snippet']['textOriginal'] for comment in response['items'][0:10]]
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

            all_comments.append(comments_in_video_info)

        except:
            # When error occurs - most likely because comments are disabled on a video
            print('Could not get comments for video ' + video_id)

    return pd.DataFrame(all_comments)


def preprocess_df(video_df):
    """
    Preprocess the df by converting dtypes, making necessary columns and utilising stopwords.
    Params:

    video_df: the created video DataFrame

    Returns:
    the preprocessed video Dataframe .

    """
    numeric_cols = ['viewCount', 'likeCount', 'commentCount']
    video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1 )
    video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x : parser.parse(x))
    video_df['publishedDay'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))
    video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
    video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')
    stop_words = set(stopwords.words('english'))
    video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
    return video_df
