import pandas as pd


def anonymize_user(user):
    first_digit = next((char for char in user if char.isdigit()), None)
    if first_digit is not None:
        return f"user_{first_digit}"
    else:
        return user


def anonymize_ad_time(ad_time):
    if ad_time <= 600:
        return 'до 600'
    else:
        return 'больше 600'


def convert_time_to_float(time_str):
    if pd.notna(time_str) and isinstance(time_str, str):
        minutes = time_str.split(':')
        total_minutes = ':'.join(minutes[:1])
        return float(total_minutes)
    else:
        return time_str


def anonymize_platform(platform):
    parts = platform.split('.')
    if len(parts) > 1:
        return '.' + parts[-1]
    else:
        return platform


def season_from_month(month):
    if 3 <= month <= 5:
        return 'весна'
    elif 6 <= month <= 8:
        return 'лето'
    elif 9 <= month <= 11:
        return 'осень'
    else:
        return 'зима'


def anonymize_ip(ip_address):
    ip_parts = ip_address.split('.')
    return '.'.join(ip_parts[:2]) + '.X.X'


def anonymize_ad_count(ad_count):
    if ad_count <= 100:
        return 'до 100'
    else:
        return 'больше 100'


def calculate_k_anonymity(data, quasi_identifiers):
    return data.groupby(quasi_identifiers, observed=False).size()


# Оценка полезности данных
def data_usefulness(original_data, anonymized_data):
    original_user_count = original_data['Пользователь'].nunique()
    anonymized_user_count = anonymized_data['Пользователь'].nunique()
    original_ip_count = original_data['IP адрес'].nunique()
    anonymized_ip_count = anonymized_data['IP адрес'].nunique()
    original_platform_count = original_data['Платформа'].nunique()
    anonymized_platform_count = anonymized_data['Платформа'].nunique()
    original_date_count = original_data['Дата просмотра'].nunique()
    anonymized_date_count = anonymized_data['Дата просмотра'].nunique()
    original_adv_count = original_data['Кол-во рекламы'].nunique()
    anonymized_adv_count = anonymized_data['Кол-во рекламы'].nunique()
    original_time_count = original_data['Время просмотра рекламы (мин)'].nunique()
    anonymized_time_count = anonymized_data['Время просмотра рекламы (мин)'].nunique()

    print("Статистика исходных данных:")
    print(f"Уникальные пользователи: {original_user_count}")
    print(f"Уникальные IP адреса: {original_ip_count}")
    print(f"Уникальные платформы: {original_platform_count}")
    print(f"Уникальная дата просмотра: {original_date_count}")
    print(f"Уникальное количество рекламы: {original_adv_count}")
    print(f"Уникальное время просмотра рекламы: {original_time_count}")

    print("\nСтатистика обезличенных данных:")
    print(f"Уникальные пользователи: {anonymized_user_count}")
    print(f"Уникальные IP адреса: {anonymized_ip_count}")
    print(f"Уникальные платформы: {anonymized_platform_count}")
    print(f"Уникальная дата просмотра: {anonymized_date_count}")
    print(f"Уникальное количество рекламы: {anonymized_adv_count}")
    print(f"Уникальное время просмотра рекламы: {anonymized_time_count}")


def main():
    input_file = 'dataset.xlsx'
    data = pd.read_excel(input_file)
    data['Дата просмотра'] = pd.to_datetime(data['Дата просмотра'], format='%d.%m.%Y')

    anonymized_data = data.copy()
    anonymized_data['Пользователь'] = data['Пользователь'].apply(anonymize_user)
    anonymized_data['IP адрес'] = data['IP адрес'].apply(anonymize_ip)
    anonymized_data['Платформа'] = data['Платформа'].apply(anonymize_platform)
    anonymized_data['Дата просмотра'] = data['Дата просмотра'].dt.month.apply(season_from_month)
    anonymized_data['Кол-во рекламы'] = data['Кол-во рекламы'].apply(anonymize_ad_count)
    data['Время просмотра рекламы (мин)'] = data['Время просмотра рекламы (мин)'].apply(convert_time_to_float)
    anonymized_data['Время просмотра рекламы (мин)'] = data['Время просмотра рекламы (мин)'].apply(anonymize_ad_time)

    # Расчет K-анонимности
    quasi_identifiers = ['Пользователь', 'IP адрес', 'Платформа', 'Дата просмотра', 'Кол-во рекламы',
                         'Время просмотра рекламы (мин)']
    k_anon_values = calculate_k_anonymity(anonymized_data, quasi_identifiers)

    # Расчет K-анонимности для исходного набора данных
    k_orig_values = calculate_k_anonymity(data, quasi_identifiers)

    print("Результаты К-анонимити для обезличенного набора данных:")
    print(k_anon_values)

    print("Результаты К-анонимити для изначального набора данных:")
    print(k_orig_values)

    data_usefulness(data, anonymized_data)

    print("Наименьшие значения К-анонимности:")
    print(k_anon_values.nsmallest(5))

    unique_rows = anonymized_data.drop_duplicates(subset=quasi_identifiers)

    if k_anon_values.min() == 1:
        print("Уникальные строки с K=1:")
        print(unique_rows)

    num_rows = len(anonymized_data)

    # Рассчитываем K-анонимность для текущего количества записей
    k_value = k_anon_values.min()

    print(f"Количество записей в датасете: {num_rows}")
    print(f"Минимальное значение K-анонимности: {k_value}")

    output_file = 'anonymized_dataset.xlsx'
    anonymized_data.drop(columns=['Вид рекламы']).to_excel(output_file, index=False)
    print(f"Анонимизированные данные сохранены в файл: {output_file}")


if __name__ == "__main__":
    main()
