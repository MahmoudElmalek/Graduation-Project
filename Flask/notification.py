import requests, aiohttp
from Flask.endpoints import url_notifications_w

# (Seaching) get req every ~sec
# f : "{person} is found!"
# x : "Can't find {person}" search is ended without finding person

#(add person) get req every ~sec
# d : "Done taking samples!" to close the camera
# n : "No new person found!"
# s : "Successfully added new person!"
# e : "error happened"


async def SendNotifications(case):
    data = {"case": case}
    async with aiohttp.ClientSession() as session:
        async with session.post(url_notifications_w, json=data) as response:
            if response.status == 200:
                print('Mobile notified',case)
            else:
                print("Failed to send notification")
