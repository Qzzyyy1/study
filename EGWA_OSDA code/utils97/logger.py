import os
import logging
import time
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def getLogger(name, args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    text = 'dataset=%s seed=%d' % (args.dataset, args.seed)
    if args.train_num != 0:
        text += ' train_num=%.2f' % args.train_num
    else:
        text += ' train_rate=%.2f' % args.train_rate

    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler('log/%s %s %s.txt' %
                             (name,
                              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                              text))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    text += ' batch=%d patch=%d epochs=%d' % (args.batch, args.patch, args.epochs)
    logger.info(name)
    logger.info(text)

    return logger

def saveFile(path, content):
    file_path, file_name = os.path.split(path)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(path, 'w') as f:
        f.write(content)

def saveJSONFile(path, save_dict, a=False):

    if a:
        with open(path, 'r') as f:
            ever = json.loads(f.read())
            save_dict = {**ever, **save_dict}

    saveFile(path, json.dumps(save_dict, indent=4))

class ProgressLogger:

    def __init__(self, epochs: int):

        self.epoch = 0

        self.progress_bar = tqdm(range(1, epochs + 1))
        self.writer = Writer()

    def update(self, dic: dict):
        self.progress_bar.update()
        self.progress_bar.set_description(f'epoch: {self.epoch + 1}')
        self.progress_bar.set_postfix(dic)
        self.writer.update(dic, self.epoch)

        self.epoch += 1

    def add_information(self, dic: dict):
        self.progress_bar.set_postfix(dic)
        self.writer.update(dic, self.epoch)

class Writer:
    
    def __init__(self):
        self.writer = SummaryWriter('logs')

    def update(self, dic: dict, epoch: int):
        for key in dic:
            self.writer.add_scalar(key, dic[key], epoch)
