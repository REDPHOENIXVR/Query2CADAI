import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DataManager(FileSystemEventHandler):
    def __init__(self, retriever):
        self.retriever = retriever
        self.observer = Observer()
        self.observer.schedule(self, path=retriever.data_dir, recursive=False)

    def on_created(self, event):
        if event.src_path.endswith('.json'):
            self.retriever.load_examples()
            self.retriever.build_index()

    def on_modified(self, event):
        if event.src_path.endswith('.json'):
            self.retriever.load_examples()
            self.retriever.build_index()

    def start(self):
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()