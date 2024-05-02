import threading
import queue
import json

class ConsumerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()
        self.running = threading.Event()

    def enqueue_data(self, data, path=None):
        data_with_path = {'data': data, 'path': path}
        self.data_queue.put(data_with_path)

    def stop(self):
        self.running.clear()

    def run(self):
        self.running.set()
        while self.running.is_set():
            try:
                data_with_path = self.data_queue.get()
            except queue.Empty:
                continue

            if data_with_path is None:
                break

            data = data_with_path['data']
            path = data_with_path['path']
            self.process_data(data, path)

            self.data_queue.task_done()

    def process_data(self, data, path=None):
        # If path is provided, write data to file
        if path:
            with open(path, 'w') as f:
                json.dump(data, f)
        else:
            # You can add alternative processing logic here
            pass
