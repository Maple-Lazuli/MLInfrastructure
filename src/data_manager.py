from dataclasses import dataclass

@dataclass
class DataManager:
    train_loader: any = None
    val_loader: any = None
    test_loader: any = None
