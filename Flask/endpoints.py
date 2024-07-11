R=12
PC=2

# RaspberryPi Server
url_R_feed =f"http://192.168.1.{R}:5000/video_feed" # Read raw feed from Raspberry
url_R_ultra = f"http://192.168.1.{R}:5000/ultra" # Read it from Raspberry

url_R_pt = f"http://192.168.1.{R}:5000/pan_tilt" # (from Ai here), write it to Raspberry
url_R_mv = f"http://192.168.1.{R}:5000/car_mv" # (from Ai here), write it to Raspberry, to SPI 

#--> may not be used as the mobile sends directly to station so to the raspberry
url_R_op = f"http://192.168.1.{R}:5000/set_opt" 



# Mobile / Station
url_mob_feed=f"http://192.168.1.{PC}:5000/vid_back" # processed feed, Write it to the station

url_frame_ids = f"http://192.168.1.{PC}:5000/frame_ids" # write it to the station
url_chs_id = f"http://192.168.1.{PC}:5000/get_chosen_id" # chosen id by mobile, read it from the station
url_chs_id_w = f"http://192.168.1.{PC}:5000/chosen_id" # where mobile writes, and AI reads from station directly
url_chs_nm_W= f"http://192.168.1.{PC}:5000/chosen_name" # mobile writes, and AI reads from station directly
url_chs_nm= f"http://192.168.1.{PC}:5000/get_chosen_name"
url_get_person=f"http://192.168.1.{PC}:5000/get_new_person"

#--> may be used  to stop the code from inside
url_opt_mob = f"http://192.168.1.{PC}:5000/set_opt" # Read it from the station 

#--> may not be used as the mobile sends directly to station so to the raspberry
url_mv_mob = f"http://192.168.1.{PC}:5000/car_mv" 


# add person
url_station_video_feed=f"http://192.168.1.{PC}:5000/video_feed"

url_notifications=f"http://192.168.1.{PC}:5000/notify"
url_notifications_w=f"http://192.168.1.{PC}:5000/notifyp"





