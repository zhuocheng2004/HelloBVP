import sys


def nogui_from_args():
    # print(sys.argv)
    return len(sys.argv) > 1 and sys.argv[1] == 'nogui'
