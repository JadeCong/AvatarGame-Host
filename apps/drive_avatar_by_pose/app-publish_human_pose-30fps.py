'''
Author: your name
Date: 2020-12-31 15:42:15
LastEditTime: 2020-12-31 16:54:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/apps/drive_avatar_by_pose/app-publish_human_pose-30fps.py
'''
import argparse
import sys, os
import time
import paho.mqtt.client as mqtt
import json
import re
import ast
from collections import OrderedDict


def parseArgs():
    parser = argparse.ArgumentParser(description='Process the argument inputs...')
    
    parser.add_argument('--broker_address', type=str, default="127.0.0.1", help="remote broker address for data transmission")
    parser.add_argument('--port', type=int, default=1883, help="monitored port for data transmission")
    parser.add_argument('--topic_pub', type=str, default="/pacific/avatar/human_keypoints_3d", help="3D human keypoints for publishing to remote broker")
    parser.add_argument('--data_path', type=str, default="../human_pose_estimation_3d/runs/detect/exp1/messages/0_msgs_pub.txt", help='path of human pose for driving avatar')
    parser.add_argument('--pub_fps', type=float, default=30, help="publish message frequency for video fps(30)")
    
    args = parser.parse_args()
    
    return args


def constructMqttClient(brokerAddress, port):
    client = mqtt.Client()
    client.connect(brokerAddress, port, 60)
    client.loop_start()
    
    return client


def publishMsgs(client, topic, message):
    # prepare the messages to be published
    strings = json.dumps(message, sort_keys= False).replace("\\", "")  # delete escape character "\"
    msgs = bytes(repr(strings).encode('utf-8'))
    
    client.publish(topic, msgs)


def mainFunc(args):
    mqtt_client = constructMqttClient(args.broker_address, args.port)
    
    try:
        with open(args.data_path, 'r') as pose_reader:
            while True:
                msg_line = pose_reader.readline()
                if msg_line != '':
                    print(msg_line)
                    values = re.search(r"OrderedDict\((.*)\)", msg_line).group(1)
                    msg_pub = OrderedDict(ast.literal_eval(values))
                    publishMsgs(mqtt_client, args.topic_pub, msg_pub)
                    time.sleep(1 / args.pub_fps)
                else:
                    break
        print("Read human pose finished.")
    finally:
        mqtt_client.disconnect()


if __name__ == '__main__':
    main_args = parseArgs()
    print("Argument Inputs: {}".format(main_args))
    print("=" * 50)
    mainFunc(main_args)
    print("=" * 50)
    print("All Done!")
