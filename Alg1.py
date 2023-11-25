import pandas as pd
import random
from datetime import datetime, timedelta


# Генерация случайного глобального IP адреса
def generate_ip():
    return f"192.104.{random.randint(1, 255)}.{random.randint(1, 255)}"


# Словарь платформ
platforms = [
    'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com', 'tiktok.com',
    'linkedin.com', 'pinterest.com', 'snapchat.com', 'reddit.com', 'vimeo.com',
    'dailymotion.com', 'whatsapp.com', 'telegram.org', 'skype.com', 'twitch.tv',
    'netflix.com', 'spotify.com', 'amazon.com', 'ebay.com', 'wikipedia.org',
    'craigslist.org', 'imdb.com', 'etsy.com', 'dropbox.com', 'flickr.com',
    'quora.com', 'soundcloud.com', 'bbc.com', 'bloomberg.com', 'forbes.com',
    'cnn.com', 'hbo.com', 'nytimes.com', 'wired.com', 'nba.com', 'nfl.com',
    'mlb.com', 'github.com', 'stackoverflow.com', 'wordpress.com', 'imgur.com',
    'espn.com', 'hulu.com', 'alibaba.com', 'aliexpress.com', 'stackoverflow.com'
]

# Словарь видов рекламы с учетом сезонов
advertisement_categories = {
    'зима': ['кроссовки Nike', 'подогревательные пледы', 'лыжное снаряжение', 'новогодние украшения', 'горячий шоколад',
             'шубы', 'грипсовые перчатки', 'ультрабуки'],
    'весна': ['цветы', 'семена для сада', 'велосипеды', 'оборудование для барбекю', 'шляпы', 'средства для аллергии',
              'спортивные костюмы', 'книги о садоводстве'],
    'лето': ['пляжные сумки', 'солнцезащитные очки', 'мороженое', 'бассейны', 'сандалии', 'шорты',
             'солнцезащитные кремы', 'велосипеды'],
    'осень': ['зонты', 'пальто', 'сапоги', 'грибы', 'книги', 'шарфы', 'теплые носки', 'подогреваемые одеяла']
}


# Генерация даты просмотра (допустим, сезон - зима)
def generate_date():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    return (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).strftime('%d.%m.%Y')


# Генерация данных для датасета, учитывая сезон
data = []
for i in range(50000):
    user = [f"example_{random.randint(1,9)}{i // 10}@mail.ru"]
    ip_address = generate_ip()
    platform = random.choice(platforms)
    date = generate_date()
    ad_count = random.randint(1, 100)
    coefficient = random.uniform(20, 360)
    ad_time = ad_count * coefficient / 60  # Преобразуем секунды в минуты

    # Определение сезона на основе месяца в дате
    month = int(date.split('.')[1])
    if month in [12, 1, 2]:
        season = 'зима'
    elif month in [3, 4, 5]:
        season = 'весна'
    elif month in [6, 7, 8]:
        season = 'лето'
    else:
        season = 'осень'

    # Выбор видов рекламы из соответствующего сезона
    advertisement_type = random.choice(advertisement_categories[season])

    data.append([user, ip_address, platform, date, ad_count, ad_time, advertisement_type])

# Создание DataFrame
df = pd.DataFrame(data, columns=['Пользователь', 'IP адрес', 'Платформа', 'Дата просмотра', 'Кол-во рекламы',
                                 'Время просмотра рекламы (мин)', 'Вид рекламы'])

# Сохранение в Excel файл с указанием полного пути
df.to_excel('C:\\Users\\kl\\Desktop\\Algorithm\\1 lab\\dataset.xlsx', index=False)


print("Датасет создан и сохранен в файл 'dataset.xlsx'")
