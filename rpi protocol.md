open terminal

'>screen'
list all screens

'>screen -ls'

'>ssh stephaniefulton@raspberrypi.local'
password: number4911

'>cd greenhouse'
'>source env/bin/activate'
'>cd e-Paper/RaspberryPi_JetsonNano/e-Paper/RaspberryPi_JetsonNano/python/examples'
'>sudo python3 screen_script.py'

can then close terminal

reattach to screen by opening terminal 
'>screen -r'
should see program running

detach
'>screen -d'


rpi troubleshooting
-after power outage, rpi has trouble reconnecting
 -turn off and on
 -if green light doesn't come on, detach hat and reattach (just red light means no connection)
