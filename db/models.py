from sqlalchemy import Table, Column, Integer, String, Float
from config.database_config import metadata

apartments = Table("apartments",
            metadata,
            Column("id", Integer, primary_key=True, index=True),
            Column("num_rooms", Integer),
            Column("district", String(50)),
            Column("floors_num", Integer),
            Column("square_total", Float),
            Column("square_living", Float),
            Column("square_kitchen", Float),
            Column("floor_cat", String(50))
)