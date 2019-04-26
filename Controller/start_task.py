import paramiko

'''
远程连接一台能ping通所有虚拟机的机器，执行上面的任务脚本
'''
def start_task():
    # 创建ssh对象
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 设置远程机器ip，端口，用户名和密码
    ssh.connect(hostname="192.168.108.217", port=22, username="eon", password="ustcipl")
    # 执行命令
    stdin, stdout, stderr = ssh.exec_command("python3 start_task.py")
    ssh.close()
