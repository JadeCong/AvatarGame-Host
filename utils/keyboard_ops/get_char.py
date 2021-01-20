'''
Author: your name
Date: 2020-12-17 16:25:57
LastEditTime: 2020-12-17 17:00:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/utils/get_char/get_char.py
'''
class _GetchUnix:
  def __init__(self):
    import tty, sys

  def __call__(self):
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print('Press a key, press q to quit')
inkey = _GetchUnix()

while True:
    print("wtf")
    k = inkey()
    print('you pressed '+k)
    if k == 'q' or k== 'Q':
        break
