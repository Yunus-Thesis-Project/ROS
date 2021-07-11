import serial

serialArduino = serial.Serial(port = "/dev/ttyUSB0", baudrate = 57600)

dataWrite = "{}{},{},{},{},{},{},{},{},{}{}{}".format("*", 90, 0, 30, 85, 0, 0, 0, 1, 1, "#","\n")

while True:
    print(dataWrite)
    serialArduino.write(dataWrite.encode())