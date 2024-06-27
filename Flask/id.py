import aiohttp
from Flask.endpoints import url_frame_ids,url_chs_id,url_chs_id_w
SearchingCond=False

async def frameIDs(current_ids):
    async with aiohttp.ClientSession() as session:
        async with session.post(url_frame_ids, json={"ids": current_ids}) as response:
            if response.status == 200:
                # print("Successfully sent current IDs to Flask server")
                pass
            else:
                print(f"Failed to share current IDs: {response.status}")

async def fetch_chosen_id():
    global chosen_id_bymobile,SearchingCond
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url_chs_id) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['id']=='s':
                        SearchingCond=True 
                    else:
                        chosen_id_bymobile = int(data['id'])
                        if chosen_id_bymobile==0: chosen_id_bymobile=None #at the end of attack, ai would post (0) to /chosen_id
                        print("Fetched chosen ID:", chosen_id_bymobile)
                else:
                    print(f"Failed to fetch chosen ID: {response.status}")
                    pass
    except Exception as e:
        print(f"Error fetching chosen ID: {e}")

async def ResetID():
    data = {"id": 0}
    async with aiohttp.ClientSession() as session:
        async with session.post(url_chs_id_w, json=data) as response:
            if response.status == 200:
                print("Reset the ID")
                pass
            else:
                print("Failed to Reset the ID")
