# Прогнозирование стоимости квартир г. Магнитогорска

## Цель проекта
Построение математической модели прогнозирования стоимости квартир г. Магнитогорска.

## Аннотация проекта
На основе данных с сайта [CityStar Магнитогорск](http://magnitogorsk-citystar.ru/realty/prodazha-kvartir) обучено несколько моделей прогнозирования стоимости квартиры (линейная регрессия, CatBoost), проведён подбор параметров моделей, их анализ и сравнение с помощью метрик MSE и R2. Исходные данные загружены в БД MySQL. Создан HTTP API-сервер на основе REST API и FastAPI. 

## Ключевые слова проекта
parsing, bs4, python, matplotlib, pandas, scikit-learn, catBoost, FastAPI, MySQL

## Описание данных

На "вход" модели подаются следующие признаки:

- id ‒ идентификатор объявления;
- num_rooms ‒ число комнат;
- flat ‒ тип планировки;
- district ‒ район расположения;
- house ‒ адрес;
- floor_num ‒ номер этажа;
- floors_num ‒ число этажей в доме;
- square_total ‒ общая площадь, м2;
- square_living ‒ жилая площадь, м2;
- square_kitchen ‒ площадь кухни, м2;

Целевой признак
- price — цена (руб.)

## Статус проекта
Завершён.