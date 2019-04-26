import socket
import sys


def oxc_send(message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('192.168.108.110', 5025))
    sock.send(message.encode('utf-8'))
    sock.settimeout(0.1)
    try:
        out = sock.recv(1024)
    except:
        out = None
    sock.close()
    return out


def test_oxc():
    message = "*idn?\n"
    print(oxc_send(message))


def oxc_swit_conn_add(inp, outp):
    message = ":oxc:swit:conn:add (@" + str(inp).strip("(").strip(")") + "), (@" + str(outp).strip("(").strip(
        ")") + ")\n"
    return oxc_send(message)


def oxc_swit_conn_port(port):
    message = ":oxc:swit:conn:port? " + str(port) + "\n"
    return oxc_send(message)


def oxc_swit_conn_stat():
    message = ":oxc:swit:conn:stat?\n"
    return oxc_send(message)

def four_type(optical_type):
#1 means three, 2 means 1-2, 3 means 1-3, 4 means 2-3
    if(optical_type==0):
        oxc_swit_conn_add((1,2,3,9,10,11),(19,25,17,18,27,26))
    elif(optical_type==1):
        oxc_swit_conn_add((1,2,3,10),(19,26,17,18))
    elif(optical_type==2):
        oxc_swit_conn_add((1,2,9,11),(25,27,17,18))
    elif(optical_type==3):
        oxc_swit_conn_add((3,10,9,11),(25,27,19,26))
