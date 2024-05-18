import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
import subprocess
import os
from manager import process

class ConsoleApp:
    def __init__(self, master):
        self.master = master
        master.title("Console Output")
        master.configure(bg="#1c1c1c")
        master.geometry("960x775")
        self.output_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=20, font=("Fixedsys", 24), fg="white", bg="#1c1c1c")
        self.output_text.pack(expand=True, fill="both")

        self.num_entry_label = tk.Label(master, text="Number of test image:", fg="white", bg="#1c1c1c")
        self.num_entry_label.pack()
        self.num_entry = tk.Entry(master)
        self.num_entry.insert(tk.END, "1")
        self.num_entry.pack()

        self.accuracy_entry_label = tk.Label(master, text="Accuracy coefficient:", fg="white", bg="#1c1c1c")
        self.accuracy_entry_label.pack()
        self.accuracy_entry = tk.Entry(master)
        self.accuracy_entry.insert(tk.END, "0.1")
        self.accuracy_entry.pack()

        self.start_button = tk.Button(master, text="Start process", command=self.start_process, fg="white", bg="#1c1c1c")
        self.start_button.pack()

        self.open_folder_button = tk.Button(master, text="Open folder", command=lambda: self.open_folder(os.path.dirname(__file__)), fg="white", bg="#1c1c1c")
        self.open_folder_button.pack()

    def start_process(self):
        num_images = int(self.num_entry.get())
        accuracy_coef = float(self.accuracy_entry.get())
        sys.stdout = self
        self.thread = threading.Thread(target=self.update_console_output, args=(num_images, accuracy_coef))
        self.thread.daemon = True
        self.thread.start()

    def update_console_output(self, num_images, accuracy_coef):
        for output_line in process(num_images, accuracy_coef):
            self.output_text.insert(tk.END, output_line)
            self.output_text.see(tk.END)

    def open_folder(self, folder_path):
        subprocess.Popen(['explorer', folder_path])

    def write(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

root = tk.Tk()
root.configure(bg="#1c1c1c")
app = ConsoleApp(root)
root.mainloop()
