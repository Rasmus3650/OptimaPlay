import threading
import queue
import json
import os

class ConsumerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()
        self.running = threading.Event()

    def enqueue_data(self, data, path=None):
        data_with_path = {'data': data, 'path': path}
        self.data_queue.put(data_with_path)

    def get_table_folder(self, folder_path, table_id):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        path = os.path.join(folder_path, f"table_{table_id}")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_game_folder(self, table_folder_path, game_id):
        path = table_folder_path + "/" + f"Game_{game_id}"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def stop(self):
        self.running.clear()

    def run(self):
        self.running.set()
        while self.running.is_set():
            try:
                data_with_path = self.data_queue.get()
            except queue.Empty:
                continue

            if "stop" in list(data_with_path['data'].keys()) and (data_with_path['data']["stop"]):
                return

            data = data_with_path['data']
            path = data_with_path['path']
            self.process_data(data, path)

            self.data_queue.task_done()

    def process_data(self, data, path):
        # If path is provided, write data to file
        cwd = os.getcwd()
        file = os.path.join(os.path.join(cwd, path), "game_data.json")
        with open(file, 'w') as f:
            f.write(json.dumps(data))