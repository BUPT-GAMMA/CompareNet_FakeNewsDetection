import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        if os.path.exists(filename):
            os.remove(filename)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def change_file(self, filename="Default.log"):
        self.log.close()
        self.log = open(filename, "a")


if __name__ == '__main__':
    sys.stdout = Logger("yourlogfilename.txt")
    print('content.')