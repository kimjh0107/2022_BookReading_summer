import yaml 

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader) # key, value로 구성되있는 형태여서 