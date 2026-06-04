import pty
import os
import time

def master_read(fd):
    return os.read(fd, 1024)

m, s = pty.openpty()

pid = os.fork()

if pid == 0:
    os.close(m)
    os.dup2(s, 0)
    os.dup2(s, 1)
    os.dup2(s, 2)
    os.close(s)
    os.execvp("python3", ["python3", "core/hardware/single_loop_field.py"])
else:
    os.close(s)
    os.write(m, b'Elysia')
    time.sleep(0.5)
    output = os.read(m, 100000).decode('utf-8')
    os.write(m, b'q')
    os.waitpid(pid, 0)

    # Just print the last few lines
    lines = output.split('\n')
    print('\n'.join(lines[-20:]))
