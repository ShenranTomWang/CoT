import json
from abc import ABC, abstractmethod
from configparser import ConfigParser

config = ConfigParser()
config.read('./config.ini')

class Dataset(ABC):
    def __init__(self, config_obj):
        self.data_path = config_obj['data_path']
        self.data = []
        self.prompt_key = config_obj['prompt_key']
        self.answer_key = config_obj['answer_key']
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __len__(self):
        return len(self.data)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

class GSM8K(Dataset):
    def __init__(self):
        super().__init__(config['GSM8K'])
    
    def load(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def save(self):
        with open(self.data_path, 'w') as f:
            for line in self.data:
                f.write(json.dumps(line) + '\n')
                
class UMWP(Dataset):
    def __init__(self):
        super().__init__(config['UMWP'])
    
    def load(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def save(self):
        with open(self.data_path, 'w') as f:
            for line in self.data:
                f.write(json.dumps(line) + '\n')

class Com2Sense(Dataset):
    def __init__(self):
        super().__init__(config['com2sense'])
    
    def load(self):
        with open(self.data_path, 'r') as f:
            json = json.loads(f)
            self.data = json["examples"]
    
    def save(self):
        with open(self.data_path, 'r') as f:
            json = json.loads(f)
        with open(self.data_path, 'w') as f:
            json["examples"] = self.data
            f.write(json.dumps(json))
            
class FantasyReasoning(Dataset):
    def __init__(self):
        super().__init__(config['fantasy_reasoning'])
    
    def load(self):
        with open(self.data_path, 'r') as f:
            json = json.loads(f)
            self.data = json["examples"]
    
    def save(self):
        with open(self.data_path, 'r') as f:
            json = json.loads(f)
        with open(self.data_path, 'w') as f:
            json["examples"] = self.data
            f.write(json.dumps(json))