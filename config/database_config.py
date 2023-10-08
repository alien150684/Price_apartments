from sqlalchemy import create_engine, MetaData

DATABASE_URL = "mysql+mysqlconnector://Aleksey_Shirokov:<YVy!_+o51hY@localhost:3306/test_db?charset=utf8mb4"

# Создание движка для подключения к базе данных
engine = create_engine(DATABASE_URL)

# Инициализация метаданных
metadata = MetaData()