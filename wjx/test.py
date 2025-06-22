from fairino import Robot
robot = Robot.RPC('192.168.58.6')
ret,version = robot.GetSDKVersion()
if ret ==0:
    print("SDK版本号为",version)
else:
    print("查询失败，错误码为",ret)