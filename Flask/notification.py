import requests
from Flask.endpoints import url_notifications

def SendNotifications(case):
    data = {"case": case}
    response = requests.post(url_notifications, json=data)
    if response.status_code == 200:
        print('Mobile notified',case)  
    else:
        print("Failed to send notification")