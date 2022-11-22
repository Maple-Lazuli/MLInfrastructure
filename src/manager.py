import threading
import time
from dataclasses import dataclass

from src.app import start
from src.trainer import Trainer


@dataclass
class Manager:
    model: any = None
    data_manager: any = None
    ip: str = "0.0.0.0"
    port: int = 5000
    epochs: int = 1
    start_watcher_app: bool = True


    def __post_init__(self):
        self.trainer = Trainer(self.model, self.data_manager, self.ip, self.port)


        self.trainer = Trainer(self.model, self.data_manager, self.ip, self.port)
        # create evaluator



    def start_watcher(self):
        threading.Thread(target=start, args=(self.ip, int(self.port), )).start()

if __name__ == "__main__":
    manager = Manager()

    manager.start_watcher()
