import aiohttp,math
import asyncio
from Flask.endpoints import url_R_mv,url_R_ultra
from Flask.id import ResetID

TIME_FOR_WHEEL_ROTATION=0.72
DISTANCE_BETWEEN_TWO_WHEELS= 20
PERIMETER_OF_WHEEL= 10

time_for_Car_rotation=((2*(22/7)*DISTANCE_BETWEEN_TWO_WHEELS)/PERIMETER_OF_WHEEL)*TIME_FOR_WHEEL_ROTATION

agreed_speed = 70

ultraF=None
ultraB=None

async def CarMoving(right,left):

    data = {"r": right, "l": left}
    async with aiohttp.ClientSession() as session:
        async with session.post(url_R_mv, json=data) as response:
            if response.status == 200:
                # print("Speeds command sent successfully", (pan_angle, tilt_angle))
                pass
            else:
                print("Failed to send Speeds command")
    #pass


async def Rotate(angle):
    # if it's positive: turn clockwise
    # if it's negative: turn counterclockwise
    tolerance=5

    if abs(angle)>= tolerance:  
        #calculate the time needed till reaching the wanted rotation angle
        rotation_time=time_for_Car_rotation/(360/angle)

        if angle > 0:
            right_speed = 0
            left_speed = agreed_speed

        elif angle < 0:
            right_speed = agreed_speed
            left_speed = 0

        await CarMoving(right_speed, left_speed)
        await asyncio.sleep(rotation_time)
        await CarMoving(0, 0)

        print('Rotated',angle)


async def CarDirectOrders(order):
        
    if order=="forward":
        print("move forward")
        await CarMoving(agreed_speed,agreed_speed)
        
    if order=="backward":
        print("move backward")
        CarMoving(-agreed_speed,-agreed_speed)

    if order=="stop":
        print("Stopped")
        CarMoving(0,0)
    
async def Attack(x,y,z):
    targeted=False
    RealDistance=z*math.cos(math.radians(y - 90))  

    # rotate till x=0
    tolerance=15
    if abs(x)>= tolerance:
        await Rotate(x-90)
        await asyncio.sleep(0.1)

    if abs(x)<= tolerance:
        targeted=True
        print('Ahead of the target')

    # attacking
    ultraSonicData = await UltraSonic()
    if z>70 and not ultraSonicData[0]:
        await CarDirectOrders("forward")
        await asyncio.sleep(0.1)
    
    if RealDistance<30 and targeted:
        await CarDirectOrders("stop")
        print("Arrived")
        ResetID()
    return 


async def UltraSonic():
    global ultraF,ultraB
    async with aiohttp.ClientSession() as session:
        async with session.get(url_R_ultra) as response:
            if response.status == 200:
                # print("Succeeded to retrieve data")
                data = await response.json()
                ultraF= False if int(data['f']) == 0 else True
                ultraB= False if int(data['b']) == 2 else True
                # print(ultraF,ultraB)
                return [ultraF,ultraB]
            else:
                print("Failed to retrieve data")
                return [False, False]

