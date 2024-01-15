import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    if not isinstance(path, str):
        raise Exception('Invalid Parameter Type')
    if path == '':
        raise Exception('Invalid File Path')
    df = pd.read_csv(path)
    return df


def data_split(data: pd.DataFrame, class_column: str):
    x = data.drop(columns=[class_column]).to_numpy()
    y = data[class_column].to_numpy()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)


