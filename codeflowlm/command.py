import subprocess

"""
def execute_command(command):
    print(command)
    result = subprocess.run(command, shell=True, text=True)
    print(result.stdout)
    if result.stderr:
      print("Error:", result.stderr)
"""
   
def execute_command(command):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Mostra a saída em tempo real
    for line in process.stdout:
        print(line, end='')  # evita pular linhas duplas

    process.wait()  # Aguarda o término