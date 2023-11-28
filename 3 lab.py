import hashlib
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def hash_phone(phone, salt):
    sha1_hash = hashlib.sha1((str(phone) + salt).encode()).hexdigest()
    md5_hash = hashlib.md5((str(phone) + salt).encode()).hexdigest()
    sha256_hash = hashlib.sha3_256((str(phone) + salt).encode()).hexdigest()
    return sha1_hash, md5_hash, sha256_hash


def save_hashes(filename, hashes):
    with open(filename, 'a') as file:
        for h in hashes:
            file.write(h + '\n')


def main():
    root = tk.Tk()
    root.title("Инструмент шифрования данных")

    file_path = None
    is_file_loaded = False
    phones = None
    numbers = None
    selected_salt_type = tk.StringVar()
    selected_salt_type.set("Numeric")

    def load_file():
        nonlocal file_path, is_file_loaded
        file_path = filedialog.askopenfilename(filetypes=[("Файлы Excel", "*.xlsx;*.xls")])
        if file_path:
            is_file_loaded = True
            button_deidentify["state"] = tk.NORMAL

    def identify():
        nonlocal file_path, phones, numbers
        df = pd.read_excel(file_path)
        hashes = df["Номер телефона"]
        numbers = [number[:-2] for number in df["Unnamed: 2"].astype(str).tolist()][:5]
        with open('hashes.txt', 'w') as f:
            for HASH in hashes:
                f.write(HASH + "\n")
        os.system("hashcat -a 3 -m 0 -o output.txt hashes.txt ?d?d?d?d?d?d?d?d?d?d?d")

        with open(r'C:\hashcat-6.2.6\output.txt') as r:
            phones = [line.strip()[33:] for line in r.readlines()]

        with open('phones.txt', 'w') as file:
            for phone in phones:
                file.write(phone + '\n')
        messagebox.showinfo("Готово", "Таблица успешно расшифрована. Данные сохранены в файле 'phones.txt'.")

    def find_salt():
        nonlocal phones, numbers
        salt_type = selected_salt_type.get()
        salts = compute_salt(phones, numbers, salt_type)
        salts = [str(salt) if isinstance(salt, int) else chr(salt) for salt in salts]
        salt_message = ", ".join(map(str, salts))
        messagebox.showinfo("Готово", f"Значение соли: {salt_message}")

    def compute_salt(phones, numbers, salt_type):
        if phones is None or not numbers:
            return []

        salts = []
        for phone in phones:
            salt = int(phone) - int(numbers[0])
            if salt < 0:
                continue

            i = 1
            while i < len(numbers) and (str(int(numbers[i]) + salt)) in phones:
                i += 1

            if i == len(numbers):
                if salt_type == "Numeric":
                    salts.append(salt + 11)
                elif salt_type == "Alphabetic":
                    salts.append(salt + ord('a'))
                elif salt_type == "Mixed":
                    salts.append(salt + ord('!'))

        return salts

    def encrypt(algorithm):
        nonlocal is_file_loaded, phones
        if not is_file_loaded:
            return
        if algorithm == "sha1":
            sha1(phones)
            messagebox.showinfo("Готово", "Результат сохранен в файле sha1.")
        elif algorithm == "sha256":
            sha256(phones)
            messagebox.showinfo("Готово", "Результат сохранен в файле sha256.")
        else:
            sha512(phones)
            messagebox.showinfo("Готово", "Результат сохранен в файле sha512.")

    def sha1(phones):
        phones_sha1 = [hashlib.sha1(phone.encode()).hexdigest() for phone in phones]
        with open('sha1.txt', 'w') as f:
            for phone in phones_sha1:
                f.write(phone + '\n')
        os.system("hashcat -a 3 -m 100 -o output_sha1.txt sha1.txt ?d?d?d?d?d?d?d?d?d?d?d")

    def sha256(phones):
        phones_sha256 = [hashlib.sha256(phone.encode()).hexdigest() for phone in phones]
        with open('sha256.txt', 'w') as f:
            for phone in phones_sha256:
                f.write(phone + '\n')
        os.system("hashcat -a 3 -m 1400 -o output_sha256.txt sha256.txt ?d?d?d?d?d?d?d?d?d?d?d")

    def sha512(phones):
        phones_sha512 = [hashlib.sha512(phone.encode()).hexdigest() for phone in phones]
        with open('sha512.txt', 'w') as f:
            for phone in phones_sha512:
                f.write(phone + '\n')
        os.system("hashcat -a 3 -m 1700 -o output_sha512.txt sha512.txt ?d?d?d?d?d?d?d?d?d?d?d")

    label_action = tk.Label(root, text="Выберите действие с таблицей:")
    label_salt_type = tk.Label(root, text="Выберите тип соли:")

    button_load = tk.Button(root, text="Загрузить файл", command=load_file)
    button_deidentify = tk.Button(root, text="Деанонимизировать", command=identify, state=tk.DISABLED)
    button_compute_salt = tk.Button(root, text="Вычислить 'соль'", command=find_salt)

    button_encrypt_sha1 = tk.Button(root, text="Зашифровать SHA-1", command=lambda: encrypt("sha1"))
    button_encrypt_sha256 = tk.Button(root, text="Зашифровать SHA-256", command=lambda: encrypt("sha256"))
    button_encrypt_sha512 = tk.Button(root, text="Зашифровать SHA-512", command=lambda: encrypt("sha512"))

    radio_numeric = tk.Radiobutton(root, text="Цифровая", variable=selected_salt_type, value="Numeric")
    radio_alphabetic = tk.Radiobutton(root, text="Буквенная", variable=selected_salt_type, value="Alphabetic")
    radio_mixed = tk.Radiobutton(root, text="Смешанная", variable=selected_salt_type, value="Mixed")

    label_action.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
    label_salt_type.grid(row=0, column=2, columnspan=2, padx=10, pady=10, sticky="w")

    button_load.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    button_deidentify.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    button_compute_salt.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    button_encrypt_sha1.grid(row=4, column=2, columnspan=2, padx=10, pady=5, sticky="e")
    button_encrypt_sha256.grid(row=5, column=2, columnspan=2, padx=10, pady=5, sticky="e")
    button_encrypt_sha512.grid(row=6, column=2, columnspan=2, padx=10, pady=5, sticky="e")

    radio_numeric.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    radio_alphabetic.grid(row=2, column=2, padx=10, pady=5, sticky="w")
    radio_mixed.grid(row=3, column=2, padx=10, pady=5, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    main()
