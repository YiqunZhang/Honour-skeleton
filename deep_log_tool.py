import requests
import io
import time
import yaml


class DeepLogTool():

    temp_logdetail_dic = {}

    def __init__(self):
        self.url = 'http://ankin.cc:50102/'
        self.session = requests.Session()
        self.log_id = "0"

    def login(self, username, password):
        url_login = self.url + 'login'
        url_check = self.url + 'checkLogin'

        params = {
            'username': username,
            'password': password
        }

        self.session.post(url_login, params)
        resp = self.session.get(url_check)
        return resp.content.decode('utf-8') == '0'

    def create_new_log(self, title, comments, dataset, task, epoch, timestamp=None):
        url = self.url + 'createNewLog'
        if timestamp is None:
            timestamp = self.timestamp()

        params = {
            'title': title,
            'comments': comments,
            'timestamp': timestamp,
            'dataset': dataset,
            'task': task,
            'epoch': epoch,
        }

        resp = self.session.get(url, params=params)
        self.log_id = resp.content.decode('utf-8')
        return len(self.log_id) == 36

    def timestamp(self):
        return int(time.time() * 1000)

    def get_yaml_text(self, config_path):
        with open(config_path) as f:
            config_yaml = yaml.load(f)

        res = ''
        for key in config_yaml.keys():
            if type(config_yaml[key]) == type({}):
                for key2 in config_yaml[key].keys():
                    if config_yaml[key][key2] != "":
                        value = config_yaml[key][key2]
                    else:
                        value = " "
                    res += f'{key}.{key2}:{value}\n'

            else:
                if config_yaml[key] != "":
                    value = config_yaml[key]
                else:
                    value = " "
                res += f'{key}:{value}\n'

        if res != "":
            res = res[:-1]

        return res

    def upload_config(self, config_path):
        config_text = self.get_yaml_text(config_path)
        url = self.url + 'uploadConfig'

        params = {
            'logId': self.log_id,
            'configText': config_text,

        }

        resp = self.session.get(url, params=params)
        resp_res = resp.content.decode('utf-8')
        return resp_res

    def upload_log_detail(self, epoch, accuracy, trainLoss, testLoss, learningRate, startTime, trainTime, testTime):
        url = self.url + 'uploadLogDetail'

        params = {
            'logId': self.log_id,
            'epoch': epoch,
            'accuracy': accuracy,
            'trainLoss': trainLoss,
            'testLoss': testLoss,
            'learningRate': learningRate,
            'startTime': startTime,
            'trainTime': trainTime,
            'testTime': testTime

        }

        resp = self.session.get(url, params=params)
        resp_res = resp.content.decode('utf-8')
        return resp_res

    def done_log(self):
        url = self.url + 'doneLog'

        params = {
            'logId': self.log_id,
        }

        resp = self.session.get(url, params=params)
        resp_res = resp.content.decode('utf-8')
        return resp_res

    def upload(self, filepath, comment):
        url = self.url + 'upload'
        files = {'file': open(filepath, 'rb')}

        data = {
            'logId': self.log_id,
            'comment': comment,
        }

        resp = self.session.post(url, files=files, data=data)
        resp_res = resp.content.decode('utf-8')
        return resp_res
