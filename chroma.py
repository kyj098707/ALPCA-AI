import chromadb
import numpy as np
import json
import pandas as pd

def initialize_chromadb():
    with open("data.json", "r", encoding="utf-8") as json_file:
        data_loaded = json.load(json_file)

    with open("synopsis_data.json", "r", encoding="utf-8") as json_file:
        synopsis_data_loaded = json.load(json_file)


    df = pd.read_csv("./csv/webtoon_data.csv", encoding="utf-8-sig")
    
    df.drop_duplicates(keep="first", inplace=True)

    data_converted_back = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in data_loaded.items()
    }

    synopsis_data_converted_back = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in synopsis_data_loaded.items()
    }

    db = chromadb.Client()
    image_collection = db.create_collection("imagestyle")
    synopsis_collection = db.create_collection("synopsis")

    is_used = []
    embeddings = []
    metadatas = []
    ids = []

    for key, val in data_converted_back.items():

        if not df.loc[df["wid"] == int(key), "title"].values:
            continue
        
        title = df.loc[df["wid"] == int(key), "title"].values[0]
        image = df.loc[df["wid"] == int(key), "thumbnail"].values[0]
        is_used.append(key)

        embeddings.append(val)
        metadatas.append({"title": title, "image":image})
        ids.append(key)

    image_collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    is_used = []
    embeddings = []
    metadatas = []
    ids = []

    for key, val in synopsis_data_converted_back.items():

        if not df.loc[df["wid"] == int(key), "title"].values:
            continue
        
        title = df.loc[df["wid"] == int(key), "title"].values[0]
        synopsis = df.loc[df["wid"] == int(key), "synopsis"].values[0]
        is_used.append(key)
        embeddings.append(val)
        metadatas.append({"title": title, "synopsis":synopsis})
        ids.append(key)

    synopsis_collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    return image_collection, synopsis_collection, df["title"].unique()




def answer(collection, title="화산귀환", task="image"):

    if task == "image":
        target_data = collection.get(where={"title": title},include=['embeddings', 'metadatas'])
        if len(target_data["embeddings"]) == 0:
            print(f"메타데이터가 '{title}'인 데이터를 찾을 수 없습니다.")
        else:
            target_embedding = target_data["embeddings"][0]
            
            similar_items = collection.query(
                query_embeddings=[target_embedding], n_results=4
            )

            return similar_items
    else:
        target_data = collection.get(where={"title": title},include=['embeddings', 'metadatas'])
        if len(target_data["embeddings"]) == 0:
            print(f"메타데이터가 '{title}'인 데이터를 찾을 수 없습니다.")
        else:
            target_embedding = target_data["embeddings"][0]
            
            similar_items = collection.query(
                query_embeddings=[target_embedding], n_results=4
            )

            return similar_items
        
        

if __name__ == "__main__":
    collection = initialize_chromadb()
    answer(collection= collection)
