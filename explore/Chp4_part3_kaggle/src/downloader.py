import os 
print(os.path.realpath(__file__))
os.chdir("/Users/kimjh/Documents/2022_BookReading_summer/explore/Chp4_part3_kaggle/")

from pathlib import Path 
import requests 
from src.config import load_config 

def download_file(data_url:str, filename:str, target_path:Path):
    r = requests.get(data_url + filename, stream=True, verify=True)
    with open(Path(target_path,filename), 'wb') as f:
        f.write(r.content)

def download_raw_data(cofig: dict[str,str], target_path: Path) -> None:
    download_file(config["url"], config["train_file"], target_path),
    download_file(config["url"], config["test_file"], target_path),

if __name__ == "__main__":

    config = load_config()
    download_raw_data(config, Path("data"))
