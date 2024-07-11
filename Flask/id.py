import aiohttp
from Flask.endpoints import url_frame_ids,url_chs_id,url_chs_id_w,url_chs_nm_W
SearchingCond=False
chosen_id_bymobile=None
chosenName_bymobile=None

async def frameIDs(current_ids):
    async with aiohttp.ClientSession() as session:
        async with session.post(url_frame_ids, json={"ids": current_ids}) as response:
            if response.status == 200:
                # print("Successfully sent current IDs to Flask server")
                pass
            else:
                print(f"Failed to share current IDs: {response.status}")



async def ResetID():
    data = {"id": 0}
    async with aiohttp.ClientSession() as session:
        async with session.post(url_chs_id_w, json=data) as response:
            if response.status == 200:
                print("Reset the ID")
                pass
            else:
                print("Failed to Reset the ID")
