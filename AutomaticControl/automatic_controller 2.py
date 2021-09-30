import tensorflow
from tensorflow_core.python.keras.saving.model_config import model_from_json

from controller import Robot
from controller import Keyboard
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow_core.python.keras.utils.np_utils import to_categorical

import os
import cv2
import struct
import numpy as np

FORWARD = ord('w')
BACKWARD = ord('s')
LEFT = ord('a')
RIGHT = ord('d')

target_counts = {
    FORWARD: 0,
    BACKWARD: 0,
    LEFT: 0,
    RIGHT: 0
}

CLASSES = ['backward', 'forward', 'left', 'right']


class PDController:

    def __init__(self, p, d, sampling_period, target=0.0):

        self.target = target
        self.response = 0.0
        self.old_error = 0.0
        self.p = p
        self.d = d
        self.sampling_period = sampling_period

    def process_measurement(self, measurement):

        error = self.target - measurement
        derivative = (error - self.old_error)/self.sampling_period
        self.old_error = error
        self.response = self.p*error + self.d*derivative

        return self.response

    def reset(self):

        self.target = 0.0
        self.response = 0.0
        self.old_error = 0.0



class MotorController():

    def __init__(self, name, robot, pd_controller):

        self.name = name
        self.robot = robot
        self.pd_controller = pd_controller
        self.motor = None
        self.velocity = 0.0

    def enable(self):
        self.motor = self.robot.getDevice(self.name)
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0.0)

    def update(self):

        self.velocity += self.pd_controller.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)

    def set_target(self, target):

        self.pd_controller.target = target

    def emergency_stop(self):

        self.motor.setVelocity(0.0)
        self.pd_controller.reset()
        self.velocity = 0.0



class Command:

    def __init__(self, left_velocity, right_velocity, emergency_stop = False):

        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.emergency_stop = emergency_stop



class Camera:

    __CHANNELS = 4

    def __init__(self, name, screenshots_folder = None):

        self.name = name
        self.frame = None
        self.image_byte_size = None
        self.device = None
        self.screenshots_folder = screenshots_folder
        self.screenshot_id = 1

    def enable(self, timestep):

        self.device = robot.getDevice(self.name)
        self.device.enable(timestep)
        self.image_byte_size = self.device.getWidth()*self.device.getHeight()*Camera.__CHANNELS

        if self.screenshots_folder is not None:
            for filename in os.listdir(self.screenshots_folder):
                if filename.endswith('.png'):
                    try:
                        screenshot_id = int(filename.split('.')[0])
                        if screenshot_id > self.screenshot_id:
                            self.screenshot_id = screenshot_id
                    except Exception:
                        pass


    def get_frame(self):

        frame = self.device.getImage()
        if frame is None:
            return self.frame

        frame = struct.unpack(f'{self.image_byte_size}B', frame)
        frame = np.array(frame, dtype=np.uint8).reshape(self.device.getHeight(), self.device.getWidth(), Camera.__CHANNELS)
        frame = frame[:,:,0:3]
        self.frame = frame
        return frame

    def show_frame(self, scale=1.5):

        scaled_frame = cv2.resize(self.frame, (0,0), fx = scale, fy = scale)
        cv2.imshow(self.name, scaled_frame)

    def save_frame(self, scale = 1.0):
        cv2.imwrite(f'{self.screenshots_folder}/{self.screenshot_id}.png', self.frame)
        self.screenshot_id += 1



FORWARD_SPEED = 12.0
TURN_SPEED = FORWARD_SPEED/2.0


automatic_commands = {
    'forward': Command(FORWARD_SPEED, FORWARD_SPEED),
    'backward': Command(-FORWARD_SPEED, -FORWARD_SPEED),
    'left': Command(-TURN_SPEED, TURN_SPEED),
    'right': Command(TURN_SPEED, -TURN_SPEED),
}

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())
timestep_seconds = timestep/1000.0

motor_left = MotorController('left wheel', robot, PDController(0.02, 0.0002, timestep_seconds))
motor_right = MotorController('right wheel', robot, PDController(0.02, 0.0002, timestep_seconds))
motor_left.enable()
motor_right.enable()

cv2.startWindowThread()
camera = Camera('kinect color', '../../screenshots')
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)


json_file = open('model_architecure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


while robot.step(timestep) != -1:

    motor_left.update()
    motor_right.update()

    image = camera.get_frame()
    predicted_move = loaded_model.predict(image)

    move = CLASSES[np.argmax(predicted_move)]

    key = keyboard.getKey()


    if move in automatic_commands.keys():
        move = automatic_commands[move]
        motor_left.set_target(move.left_velocity)
        motor_right.set_target(move.right_velocity)


    if (key == ord('E')):
        motor_left.emergency_stop()
        motor_right.emergency_stop()


    cv2.waitKey(timestep)
cv2.destroyAllWindows()