import pandas as pd
import shutil
from glob import glob
import json


def modify_image_path():
    folder_path = sorted(glob(f'./image/*'))
    for f in folder_path:
        cnt = 1
        img_path = sorted(glob(f+'/*.jpg'))
        for img in img_path:
            shutil.move(img,f+f'/train_{cnt}.jpg')
            cnt += 1

def get_comic_up_20():
    """
    20회차 이상의 웹툰
    """
    folder_path = sorted(glob(f'./image/*'))
    comic_list = []
    for f in folder_path:
        img_path = sorted(glob(f+'/*.jpg'))
        if len(img_path) == 20:
            comic_list.append(f.split("\\")[-1])
   
    return comic_list


def make_train_csv(comic_list, comic_df):
    cnt = 0
    for i in range(len(comic_list)):
        print(i)
        img_list = glob(f'./image/{comic_list[i]}/*.jpg')
        input_label = [0 for _ in range(len(comic_list))]
        input_label[i] = 1
        _input = {}
        for img in img_list:
            print(img)
            _input['img_path'] = img
            for k,v in zip(comic_list,input_label):
                _input[k] = int(v)
            comic_df.loc[cnt] = _input
            cnt += 1
            
    comic_df.to_csv("./comic.csv")

def synopsis_to_json():
    def str_to_array(embedding_str):
        return [float(emb.strip()) for emb in embedding_str[1:-1].split(" ") if emb != ""]
    
    df = pd.read_csv("./synopsis.csv", encoding="utf-8-sig")
    df['embedding'] = df['embedding'].apply(str_to_array)
    dic = dict(zip(df["id"], df["embedding"]))
        
    with open("synopsis_data.json", 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    synopsis_to_json()
    """
    comic_list = get_comic_up_20()
    comic_df = pd.DataFrame(columns=["img_path",*comic_list])
    make_train_csv(comic_list,comic_df)
    """