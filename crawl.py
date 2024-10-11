"""
크롤링 CODE
"""

import os
import requests
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv


def get_settings():
    load_dotenv()
    return (
        os.environ.get("WEBTOON_URL"),
        os.environ.get("WEBTOON_DETAIL_URL"),
        os.environ.get("WEBTOON_INFO_URL"),
        {"User-Agent": os.environ.get("AGENT")},
    )


def crawl_data(
    save_image,
    save_info,
    make_excel,
    headers,
    webtoon_home_url,
    webtoon_episode_url,
    webtoon_info_url,
):
    response = requests.get(webtoon_home_url)
    if response.status_code != 200:
        raise ()

    data = response.json()
    webtoon_data = []
    for day, webtoon_list in data["titleListMap"].items():
        for webtoon in webtoon_list:
            webtoon_title = ""
            webtoon_synopsis = ""
            webtoon_artist = ""
            wid = webtoon["titleId"]
            os.makedirs(f"./image/{wid}", exist_ok=True)
            detail_webtoon_url = webtoon_episode_url + str(wid)
            response = requests.get(detail_webtoon_url)
            detail_data = response.json()

            if detail_data == "LOGIN":
                continue  # 연령제한 content skip

            episode_list = detail_data["articleList"]

            if save_image:
                """
                webtoon_image
                """
                for idx, episode in enumerate(episode_list):
                    img_response = requests.get(
                        episode["thumbnailUrl"], headers=headers
                    )
                    if img_response.status_code != 200:
                        continue
                    with open(f"image/{wid}/{idx+1}.jpg", "wb") as file:
                        file.write(img_response.content)

            if save_info:
                """
                webtoon_artist, webtoon_title, webtoon_synopsis
                """
                info_webtoon_url = webtoon_info_url + str(wid)
                response = requests.get(info_webtoon_url)
                data = response.json()
                for artist_info in data["communityArtists"]:
                    if "ARTIST_PAINTER" in artist_info["artistTypeList"]:
                        webtoon_artist = artist_info["artistId"]
                        break
                webtoon_title = data["titleName"]
                webtoon_synopsis = data["synopsis"]
                webtoon_url = data["thumbnailUrl"]

            if make_excel:
                """
                result: ./csv/webtoon_data.csv
                """
                webtoon_data.append(
                    {
                        "wid": wid,
                        "title": webtoon_title,
                        "synopsis": webtoon_synopsis,
                        "artist": webtoon_artist,
                        "thumbnail" : webtoon_url
                    }
                )

    df = pd.DataFrame(webtoon_data)
    df.to_csv("./csv/webtoon_data.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=bool, default=False)
    parser.add_argument("--info", type=bool, default=False)
    parser.add_argument("--excel", type=bool, default=False)
    home_url, detail_url, info_url, headers = get_settings()

    args = parser.parse_args()

    crawl_data(
        save_image=args.image,
        save_info=args.info,
        make_excel=args.excel,
        headers=headers,
        webtoon_home_url=home_url,
        webtoon_episode_url=detail_url,
        webtoon_info_url=info_url,
    )
