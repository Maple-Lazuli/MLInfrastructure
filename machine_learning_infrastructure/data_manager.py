from dataclasses import dataclass

@dataclass
class DataManager:
    train_loader: any = None
    validation_loader: any = None
    test_loader: any = None
    classes: any = None
