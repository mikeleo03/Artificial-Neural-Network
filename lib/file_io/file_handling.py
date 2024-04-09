import json

# Untuk membaca file json yang menjadi masukan
def load_json(filename):
    with open(f"test/{filename}", "r") as file:
        data = json.load(file)
    return data

# Menyimpan file json
def save_json(data, filename):
    with open(f"test/{filename}", "w") as file:
        json.dump(data, file, indent=4)