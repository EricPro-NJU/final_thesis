import requests

token = "cabb7bd0508a"


def send_message(title, name, content, token=token):
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "token",
                             "title": "title",
                             "name": "name",
                             "content": "content"
                         })
    print(resp.content.decode())


def send_log_message(task_name, log_str, token=token):
    title = "任务{}从AutoDL服务器发来消息".format(task_name)
    send_message(title, task_name, log_str, token)
