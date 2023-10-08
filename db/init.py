# Импортируем необходимые модули и метаданные
from config.database_config import engine
from models import metadata

def init_database():
    """
    Функция для инициализации базы данных. Создает таблицы на основе метаданных.
    """
    metadata.create_all(bind=engine)

# Если этот файл запущен как основной, инициализируем базу данных
if __name__ == "__main__":
    init_database()