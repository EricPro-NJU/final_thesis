import requests


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
