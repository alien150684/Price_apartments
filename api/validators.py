from pydantic import BaseModel
from enum import Enum

class District(str, Enum):
    ORDZHONIKIDZE = 'орджоникидзевский'
    LENINSKY = 'ленинский'
    PRAVOBEREJNY = 'правобережный'

class Apartment(BaseModel):
    num_rooms: int
    district: District
    floors_num: int
    square_total: float
    square_living: float
    square_kitchen: float
    floor_cat: str