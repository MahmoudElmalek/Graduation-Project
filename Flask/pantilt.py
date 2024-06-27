import aiohttp
from Flask.endpoints import url_R_pt
async def PanTiltMoving(pan_angle,tilt_angle):
    data = {"pan": pan_angle, "tilt": tilt_angle}
    async with aiohttp.ClientSession() as session:
        async with session.post(url_R_pt, json=data) as response:
            if response.status == 200:
                # print("Pan-Tilt command sent successfully", (pan_angle, tilt_angle))
                pass
            else:
                print("Failed to send Pan-Tilt command")
    # pass

