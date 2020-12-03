# Проект "Прогнозирование стоимости автомобилей v2"

## Цели и задачи проекта

Данный проект является продолжением проекта по предсказанию стоимости автомобилей.
Теперь кроме табличных данных мы можем проанализировать текст с объявлением и 
фотографии автомобилей и применить эту информацию для улучшения работы модели.

### В данном проекте мы применили знания, полученные в юнитах по Deep Learning: 
1) применение сверточных нейронных сетей для работы с табличными данными
2) анализ изображений
3) NLP
4) создание multi-input сетей

### В ходе работы над проектом были решены следующие задачи:

→ EDA

    Датасет очищен от дубликатов, пропущенных значений.
    Подобраны новые переменные, выполнена нормализация и стандартизация признаков, применены методы для кодирования категориальных признаков.
    Произведена генерация новых признаков и отбор признаков методом Стьюдента и при помощи коррекляционной матрицы.

→ Построение модели по обработке естественного языка (NLP)

    Применены стандартные архитектуры.
    Обработка текста: применены дополнительные методы предобработки (токенизация, лемматизация).

→ Добавление изображений в решение

    Применены стандартные архитектуры.
    Применены различные методы обучения (Fine-tuning, transfer-learning, LR-Cycle).
    Обработка изображений: применены дополнительные методы предобработки (аугментация).
    Применены продвинутые архитектуры (SOTA) для обработки изображений.

→ Обучение модели:

    Построена «наивная» ML-модель на табличных данных.
    Модель градиентного бустинга с помощью CatBoost.
    DL (модель NN Tabular)
    Multi-Input нейронная сеть для анализа табличных данных и текста
    Multi-Input нейронная сеть для анализа табличных данных, текста и изображений
    Multi-Input нейронная сеть с пробросом признака
    усреднение предсказаний


## Информация о данных

Все данные, которые использовались в проекте, были взяты с соревнования на Kaggle.
На вход подавались датасет с различными признаками, тексты объявлений и изображения автомобилей.

## Файлы в репозитории

    car_price_part2_best_final.ipynb - командный проект
    nlp.py - модуль поиска самых повторяющихся фраз в описаниях автомобилей.

Никнейм на Kaggle - Alexander Samokhin  
Значение метрики, которого удалось добиться - 10.78708
