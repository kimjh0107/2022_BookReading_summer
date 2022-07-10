import os 
os.chdir("/Users/kimjh/Documents/2022_BookReading_summer/explore/Chp4_part3_kaggle/")

from pathlib import Path
import pandas as pd
import torch


def read_csv_files(train_data_path:Path, test_data_path:Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_data_path), pd.read_csv(test_data_path)

def merge_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]]) # train에만 있는 값 제거 

def get_numeric_column_index(df: pd.DataFrame) -> pd.Index: # numerical data 처리를 위한 pd.Index
    return df.dtypes[df.dtypes != 'object'].index 

def normalize_numeric_columns(df:pd.DataFrame) -> pd.DataFrame:
    numeric_columns = get_numeric_column_index(df)
    df[numeric_columns] = df[numeric_columns].apply(lambda x:(x-x.mean()) / x.std())
    return df 

def fill_null_values(df: pd.DataFrame):
    numeric_columns = get_numeric_column_index(df)
    df[numeric_columns] = df[numeric_columns].fillna(0)
    return df 

def dummify_columns(df:pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, dummy_na=True)

def create_feature_tensor(data, n_train: int) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(data[:n_train].to_numpy(), dtype=torch.float32)
    test_X = torch.tensor(data[n_train:].to_numpy(), dtype=torch.float32)
    return train_X, test_X

def create_label_tensor(train_df:pd.DataFrame) -> torch.Tensor:
    return torch.tensor(train_df.iloc[:, -1].to_numpy().reshape(-1,1), dtype=torch.float32) # 여기서 reshape(-1)은 numpy를 한줄로, reshape(-1,1) 로 변경을 해주면 한줄 한줄 각각에 대해서 labels값 입력 

if __name__ == "__main__":
    train_df, test_df = read_csv_files(
        Path("data/kaggle_house_pred_train.csv"),
        Path("data/kaggle_house_pred_test.csv"),
    )

    data = (
        merge_train_test_data(train_df, test_df)
        .pipe(normalize_numeric_columns)
        .pipe(fill_null_values)
        .pipe(dummify_columns)
    )

    n_train = train_df.shape[0]
    train_X, test_X = create_feature_tensor(data, n_train)
    train_y = create_label_tensor(train_df)

    torch.save(train_X, "data/processed/train_X.pt")
    torch.save(test_X, "data/processed/test_X.pt")
    torch.save(train_y, "data/processed/train_y.pt")


