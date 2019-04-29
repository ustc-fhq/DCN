import random
import subprocess


def single_task(task_type, task_size, task_id):
    """
    :param task_size: 任务的大小：三个等级：0：没有任务；1：持续时间在[0,30]秒的小任务；2：持续时间在[80, 100]秒的大任务
    :param task_type: ‘t’-型：I/O密集型任务，否则，CPU密集型任务
    :param task_id: 任务的id，最终根据id生成log文件
    :return:
    """
    # 确定任务大小
    task_size = int(task_size)
    task_size=random.normalvariate(task_size, task_size/20)
    # 确定任务生成文件和log文件
    task_file = "/benchmark/" + str(task_id)
    log_file = "hadoop_log/" + str(task_id) + ".log"
    if task_type == 't':
        output = subprocess.getoutput('hadoop jar /opt/hadoop/share/hadoop/mapreduce'
                                          '/hadoop-mapreduce-client-jobclient-2.9.2-tests.jar '
                                          'TestDFSIO -write -nrFiles 2 -size 2000MB')
            file.write('finish' + str(datetime.datetime.now()))
    else:
        print('cpu job')
        # task_large=int(task_large*1000000)
        output = subprocess.getoutput('hadoop jar '
                                      '/opt/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.9.2.jar'
                                      ' pi 40 1000000  >>logs/job-' + str(job_num) + '.txt')
        print(output)
