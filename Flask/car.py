import aiohttp,math
import asyncio
from Flask.endpoints import url_R_mv,url_R_ultra
from Flask.id import ResetID

RPM=4000
PERIMETER_OF_WHEEL= .38
TIME_FOR_WHEEL_ROTATION=1/((RPM/60)*PERIMETER_OF_WHEEL)
DISTANCE_BETWEEN_TWO_WHEELS= .38

ultraF=False
ultraB=False

# for full 360 around the car
# time_for_Car_rotation=((2*(22/7)*DISTANCE_BETWEEN_TWO_WHEELS)/PERIMETER_OF_WHEEL)*TIME_FOR_WHEEL_ROTATION
# time_for_Car_rotation=22*3.1/7

# agreed_speed_r = 90
# agreed_speed_l = 190


# async def CarMoving(right,left):

#     data = {"r": right, "l": left}
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url_R_mv, json=data) as response:
#             if response.status == 200:
#                 # print("Speeds command sent successfully", (pan_angle, tilt_angle))
#                 pass
#             else:
#                 print("Failed to send Speeds command")
#     #pass


# async def Rotate(angle):
#     # if it's positive: turn clockwise
#     # if it's negative: turn counterclockwise
#     tolerance=5

#     if abs(angle)>= tolerance:  
#         #calculate the time needed till reaching the wanted rotation angle
#         rotation_time=time_for_Car_rotation*(angle/360)

#         if angle > 0:
#             right_speed = 0
#             left_speed = agreed_speed_l

#         elif angle < 0:
#             right_speed = agreed_speed_r
#             left_speed = 0

#         await CarMoving(right_speed, left_speed)
#         await asyncio.sleep(rotation_time)
#         await CarMoving(0, 100) # stopping

#         print('Rotated',angle)


# async def CarDirectOrders(order):
        
#     if order=="forward":
#         print("move forward")
#         await CarMoving(agreed_speed_r,agreed_speed_l)
        
#     if order=="backward":
#         print("move backward")
#         await CarMoving(35,135)

#     if order=="stop":
#         print("Stopped")
#         await CarMoving(1,101)
    
# async def Attack(xDelta, yDelta, zTrack):
#     tolerance=15

#     await UltraSonic()

#     if zTrack>95 and not ultraF :
#         if xDelta>tolerance:
#             print('steering right')
#             agreed_speed_r=-xDelta/20
#             print(agreed_speed_r,agreed_speed_l)
#             # await CarMoving(agreed_speed_r,agreed_speed_l)
#         elif xDelta<-tolerance:
#             print('steering left')
#             agreed_speed_l=-xDelta/20
#             print(agreed_speed_r,agreed_speed_l)
#             # await CarMoving(agreed_speed_r,agreed_speed_l)
#         else:
#             print("Ahead")
#             # await CarDirectOrders("forward")
#     else:
#             # await CarDirectOrders("stop")
#             print("Arrived")
#             await ResetID()
#     return 

class CarController:
    def __init__(self, rpm, wheel_perimeter, distance_between_wheels):
        self.rpm = rpm
        self.wheel_perimeter = wheel_perimeter
        self.wheel_distance = distance_between_wheels
        self.agreed_speed_r = 90
        self.agreed_speed_l = 190
        self.ultraF = False
        self.ultraB = False
        self.tolerance = 15

        self.time_for_wheel_rotation = 1 / ((self.rpm / 60) * self.wheel_perimeter)
        # self.time_for_car_rotation = ((2 * (22 / 7) * self.wheel_distance) / self.wheel_perimeter) * self.time_for_wheel_rotation
        self.time_for_car_rotation = 22 * 3.1 / 7

    async def rotate(self, angle):
        tolerance = 5
        if abs(angle) >= tolerance:
            rotation_time = self.time_for_car_rotation * (angle / 360)
            if angle > 0:
                right_speed = 1
                left_speed = 190
            elif angle < 0:
                right_speed = 90
                left_speed = 1

            await self.car_moving(right_speed, left_speed)
            await asyncio.sleep(rotation_time)
            await self.car_moving(1, 101)  # stopping

            print('Rotated', angle)

    async def car_direct_order(self, order):
        if order == "forward":
            print("move forward")
            await self.car_moving(95, 195)
        elif order == "backward":
            print("move backward")
            await self.car_moving(45, 145)
        elif order == "stop":
            print("Stopped")
            await self.car_moving(1, 101)

    async def attack(self, xDelta, zTrack):
        if zTrack > 95 and not self.ultraF:
            if xDelta > self.tolerance:
                print('steering right')
                self.agreed_speed_l = max(185, self.agreed_speed_l - (xDelta / 40))
                print("speeds",self.agreed_speed_r, self.agreed_speed_l)
                # await self.car_moving(self.agreed_speed_r, self.agreed_speed_l)
            elif xDelta < -self.tolerance:
                print('steering left')
                self.agreed_speed_r = max(85, self.agreed_speed_r - (abs(xDelta) / 40))
                print("speeds",self.agreed_speed_r, self.agreed_speed_l)
                # await self.car_moving(self.agreed_speed_r, self.agreed_speed_l)
            else:
                print("Ahead")
                # await self.car_direct_order("forward")

            # await asyncio.sleep(0.1)
            # await self.ultra_sonic()
        # if self.ultraF:
        #     print("Obstacle ahead! Stopping.")
            # await self.car_direct_order("stop")
            # return
        elif zTrack<=95:
            print("Arrived at the target")
            # await self.car_direct_order("stop")
            self.agreed_speed_r = 90
            self.agreed_speed_l = 190
            await ResetID()

    async def car_moving(self, right, left):
        data = {"r": right, "l": left}
        async with aiohttp.ClientSession() as session:
            async with session.post(url_R_mv, json=data) as response:
                if response.status == 200:
                    pass
                else:
                    print("Failed to send Speeds command")

    async def ultra_sonic(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(url_R_ultra) as response:
                if response.status == 200:
                    data = await response.json()
                    self.ultraF = False if int(data['f']) == 0 else True
                    self.ultraB = False if int(data['b']) == 2 else True
                    return [self.ultraF, self.ultraB]
                else:
                    print("Failed to retrieve ultra values")
                    return [False, False]

RobotControl = CarController(RPM, PERIMETER_OF_WHEEL, DISTANCE_BETWEEN_TWO_WHEELS)
