import pandas as pd
from config.database_config import engine
from models import apartments

def insert_dataframe_to_db(df, table_name):
    """
    Функция для загрузки данных из DataFrame в базу данных.
    
    Параметры:
    - df (DataFrame): DataFrame с данными для загрузки.
    - table_name (str): Имя таблицы в базе данных, в которую будут загружены данные.
    """
    df.to_sql(table_name, con=engine, if_exists='append', index=False)

if __name__ == "__main__":
    # Чтение данных из CSV-файла в DataFrame
    df = pd.read_csv('data\CityStarExport-04.10.2023.csv')
    
    # Загрузка данных из DataFrame в базу данных
    insert_dataframe_to_db(df, 'apartments')