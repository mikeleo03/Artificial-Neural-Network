class JSONReader:
    """
    Melakukan pembacaan kasus uji dalam format JSON

    Atribut:
        filename: Nama fail JSON yang dibaca.
        data: Isi dari fail JSON yang dibaca.
        learning_rate: Learning rate dari model
    """
    
    # Inisiasi kelas
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None

    # Membaca data JSON
    def read(self):
        with open(self.filename, 'r') as f:
            self.data = json.load(f)
        return self.data
    