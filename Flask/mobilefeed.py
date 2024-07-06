import aiohttp,cv2,requests
from Flask.endpoints import url_mob_feed

async def MobileFeed(feed):
    # Encode the frame into JPEG format
        _, encoded_frame = cv2.imencode('.jpg', feed)

        # Send the encoded frame as a POST request to the server
        response = requests.post(url_mob_feed, data=encoded_frame.tobytes(), headers={'Content-Type': 'image/jpeg'})

        # if response.status_code == 200:
        #     # print('done sendnig frames')
        #     pass
        # else:
        #     print("Failed to send frames")
        # pass
