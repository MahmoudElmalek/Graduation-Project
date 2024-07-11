import aiohttp
from Flask.endpoints import url_R_pt

class PanTiltController:
    def __init__(self):
        self.pan_angle = 0
        self.tilt_angle = 45

    async def move(self, xDelta, yDelta):
        self.pan_angle += xDelta // 45
        self.tilt_angle += yDelta // 45

        self.pan_angle = max(-90, min(90, self.pan_angle))
        self.tilt_angle = max(5, min(90, self.tilt_angle))

        print(self.pan_angle,self.tilt_angle)
        
        data = {"pan": self.pan_angle, "tilt": self.tilt_angle}
        async with aiohttp.ClientSession() as session:
            async with session.post(url_R_pt, json=data) as response:
                if response.status == 200:
                    pass
                else:
                    print("Failed to send Pan-Tilt command")
    

    async def ytracking(self, xDelta, yDelta):
        self.pan_angle = 0
        self.tilt_angle += yDelta // 45

        self.tilt_angle = max(5, min(90, self.tilt_angle))

        print(self.pan_angle,self.tilt_angle)
        
        data = {"pan": self.pan_angle, "tilt": self.tilt_angle}
        async with aiohttp.ClientSession() as session:
            async with session.post(url_R_pt, json=data) as response:
                if response.status == 200:
                    pass
                else:
                    print("Failed to send Pan-Tilt command")
        # return self.pan_angle

    async def order(self, pan, tilt):

        self.pan_angle = max(-90, min(90, pan))
        self.tilt_angle = max(5, min(90, tilt))

        print(self.pan_angle,self.tilt_angle)
        
        data = {"pan": self.pan_angle, "tilt": self.tilt_angle}
        async with aiohttp.ClientSession() as session:
            async with session.post(url_R_pt, json=data) as response:
                if response.status == 200:
                    pass
                else:
                    print("Failed to send Pan-Tilt command")

PanTiltMoving = PanTiltController()
