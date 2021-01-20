'''
Author: your name
Date: 2020-12-17 17:27:28
LastEditTime: 2020-12-17 20:32:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/utils/keyboard_ops/key_pressed.py
'''
import sys, tty, termios, select
import time


class GetchUnix:
    def __init__(self):
        self.fd = sys.stdin
        self.when = termios.TCSADRAIN
        self.old_settings = termios.tcgetattr(self.fd)
    
    def _isData(self):
        data_flag = select.select([self.fd], [], [], 0) == ([self.fd], [], [])
        
        return data_flag
    
    def __call__(self, key):
        if key.islower():
            key.upper()  # check the upper char standard
        
        try:
            tty.setcbreak(self.fd.fileno())
            if self._isData():
                char = self.fd.read(1)
                if char.islower():
                    char.upper()
                if char == key:
                    return True
        finally:
            termios.tcsetattr(self.fd, self.when, self.old_settings)
        
        return False


check_key = GetchUnix()


def is_pressed(key):
    key_flag = check_key(key)
    
    return key_flag
