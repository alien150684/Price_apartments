import requests

URL = 'http://127.0.0.1:8000/predict/'

# Создаем список с различными данными для отправки
data_list = [
    {
        'num_rooms': 2,
        'district': 'правобережный',
        'floors_num': 5,
        'square_total': 42.0,
        'square_living': 30.0,
        'square_kitchen': 6.0,
        'floor_cat': 'верхний'
    },
    {
        'num_rooms': 3,
        'district': 'орджоникидзевский',
        'floors_num': 9,
        'square_total': 74.0,
        'square_living': 0.0,
        'square_kitchen': 10.0,
        'floor_cat': 'промежуточный'
    },
    {
        'num_rooms': 1,
        'district': 'орджоникидзевский',
        'floors_num': 10,
        'square_total': 30.2,
        'square_living': 20.0,
        'square_kitchen': 5.0,
        'floor_cat': 'промежуточный'
    }
]

# Отправляем каждый набор данных в отдельном запросе
for data in data_list:
    response = requests.post(URL, json=data)
    print(response.text)