import os
import sys

def update_sys_path():
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))

update_sys_path()

