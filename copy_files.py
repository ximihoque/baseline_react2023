import os
from tqdm import tqdm
import sys
from scp import SCPClient
import paramiko

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient('172.30.1.197', '22', 'surbhi', 'surbhi')
scp = SCPClient(ssh.get_transport())

def read_todo(todo_path):
    """Reads TODO file to process videos
    """
    try:
        with open(todo_path) as f:
            d = [i.strip('\n') for i in f.readlines()]
        return d
    except:
        print ("File not found!")
        return []


def files_to_process(todo_path):
    processed_todo_path = todo_path.split(".txt")[0] + '_processed.txt'
    
    todo = read_todo(todo_path)
    proc = read_todo(processed_todo_path)
    
    d = list(set(todo) - set(proc))
    return d

def write_processed(todo_path, fname):
    """Writes processed video[s] here
    """
    processed_todo_path = todo_path.split(".txt")[0] + '_processed.txt'
    with open(processed_todo_path, 'a') as f:
        f.write(fname)
        f.write('\n')

def write_errors(todo_path, fname):
    processed_todo_path = todo_path.split(".txt")[0] + '_errors.txt'
    with open(processed_todo_path, 'a') as f:
        f.write(fname)
        f.write('\n')

def main(todo_path):
    files = files_to_process(todo_path)
    print ('files to process: ', len(files))
    for todo in tqdm(files):
        src = '/home/surbhi/ximi/REACT/main/' + todo.split('../')[-1]
        dst = todo.split('../')[-1]
        scp.get(src, dst, recursive=True)
        write_processed(todo)

if __name__ == '__main__':
    args = sys.argv
    main(args[1])