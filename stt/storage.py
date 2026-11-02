"""Sauvegarde des transcriptions."""

import os
import time


def save_text(text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_name = time.strftime("%Y%m%d_%H%M%S") + ".txt"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text.strip())
    return file_path
