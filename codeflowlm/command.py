import subprocess

def execute_command(command):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)  

    process.stdout.close()
    process.wait()