import requests
import time


def send_message(title, name, content):
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "cabb7bd0508a",
                             "title": title,
                             "name": name,
                             "content": content
                         })
    print(resp.content.decode())


def send_log_message(task_name, log_str):
    title = "任务{}从AutoDL服务器发来消息".format(task_name)
    send_message(title, task_name, log_str)


if __name__ == "__main__":
    send_log_message("test", "这是一条测试消息。哈哈哈哈哈~~~")


class Log:
    def __init__(self, task_name):
        self.task_name = task_name
        self.log_list = []
        self.log_path = "/root/autodl-nas/log/{}_{}.log".format(task_name,
                                                                time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.log_num = 0

    def log(self, text, mute=False, discard=False, message=False):
        log_text = "{}\t{}".format(time.strftime("LOG: %Y-%m-%d %H:%M:%S", time.localtime()), text)
        if not discard:
            self.log_list.append("{}\n".format(log_text))
            self.log_num += 1
        if not mute:
            print(log_text)
        if message:
            send_log_message(self.task_name, text)

    def writelog(self):
        if self.log_num == 0:
            print("Warning from class Log: No log will be written.")
            return
        with open(self.log_path, "w", encoding="UTF-8") as fp:
            fp.writelines(self.log_list)
        self.log_list.clear()
        self.log_num = 0
        self.log_path = "/root/autodl-nas/log/{}_{}.log".format(self.task_name,
                                                                time.strftime("%Y%m%d%H%M%S", time.localtime()))
