import adafruit_sht31d
import board
import time
import busio
import adafruit_veml7700
import datetime
import os
import warnings
from RPi import GPIO
import csv
#from adafruit_extended_bus import ExtendedI2C as I2C
warnings.filterwarnings("ignore", category=RuntimeWarning)

#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import os

picdir = '/home/stephaniefulton/greenhouse/e-Paper/RaspberryPi_JetsonNano/e-Paper/RaspberryPi_JetsonNano/python/pic'
libdir = '/home/stephaniefulton/greenhouse/e-Paper/RaspberryPi_JetsonNano/e-Paper/RaspberryPi_JetsonNano/python/lib'
if os.path.exists(libdir):
   sys.path.append(libdir)
#print(picdir)

import logging
from waveshare_epd import epd4in2_V2
import time
from PIL import Image,ImageDraw,ImageFont
import traceback

logging.basicConfig(level=logging.DEBUG)

i2c = board.I2C() #i2c for humidity sensor
sensor = adafruit_sht31d.SHT31D(i2c)
cel = sensor.temperature
fah = (1.8 * cel) + 32
tempText = 'temperature'
temp = tempText + ': ' + str(fah)
#print(fah)
#print(temp)
new = '%.2f' % sensor.relative_humidity
#print(str(new))
#gather light and lux
from adafruit_extended_bus import ExtendedI2C as I2C
#specify i2c port for veml
i2c_4 = I2C(5)
veml7700 = adafruit_veml7700.VEML7700(i2c_4)
light = '%.2f' % veml7700.light
#print("Ambient light:", veml7700.light)
#print("Lux:", veml7700.lux)
#print('\u273d')

try:
    logging.info("epd4in2 Demo")
    
    epd = epd4in2_V2.EPD()
    logging.info("init and Clear")
    epd.init()
    epd.Clear()
    
    font24 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 24)
    font18 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 18)
    font35 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 35)
    font55 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 55)
    font85 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 85)
    bmpFan = Image.open(os.path.join(picdir, 'fan.bmp'))
    bmpLight = Image.open(os.path.join(picdir, 'light.bmp'))
    print(font24)
    
    current = datetime.datetime.now()
    fileDay = current.strftime("%y%m%d")
    fileTime = current.strftime("%H%M%S")
    fileTime = "/home/stephaniefulton/greenhouse/" + fileDay + "-" + fileTime + ".csv"
    with open(fileTime, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "humidity", "temperature", "lux", "ambient", "fan"])
    
    if 0:
        logging.info("E-paper refresh")
        epd.init()
        # Drawing on the Horizontal image
        logging.info("1.Drawing on the Horizontal image...")
        num = 0
        while(True):
            #pull from sensors every loop
            now = datetime.datetime.now()
            cel = sensor.temperature
            fah = (1.8 * cel) + 32
            fah = '%.1f' % fah

            temp = str(fah) + '\xb0' + 'F'
            humidity = '%.1f' % sensor.relative_humidity
            hum = str(humidity) + '%'
            light = '%.f' % veml7700.light
            #light = str(light)
            lux = '%.f' % veml7700.lux
            #lux = str(lux)

            
            Himage = Image.new('1', (epd.width, epd.height), 255)  # 255: clear the frame
            draw = ImageDraw.Draw(Himage)
            draw.text((10, 150), hum, font = font55, fill = 0)
            draw.line((190, 160, 190, 300), fill = 0)
            draw.text((200, 150), temp, font = font55, fill = 0)
            draw.text((0, 10), time.strftime('%I:%M'), font = font85, fill = 0)
            draw.text((215, 65), time.strftime('%p'), font = font24, fille = 0)
            draw.text((0, 95), time.strftime('%A  %D'), font = font35, fille = 0)
            draw.line((0, 150, 400, 150), fill = 0)
            if (sensor.relative_humidity > 50) and (veml7700.light < 10000) :
                os.system('sudo uhubctl -a on -l 2 > /dev/null')
                Himage.paste(bmpFan, (380,10))
                fan = "on"
            if (sensor.relative_humidity < 50) and (veml7700.light < 10000):
                os.system('sudo uhubctl -a off -l 2 > /dev/null')
                fan = "off"
            if (sensor.relative_humidity > 50) and (veml7700.light > 10000):
                os.system('sudo uhubctl -a on -l 2 > /dev/null')
                Himage.paste(bmpFan, (380,10))
                fan = "on"
                Himage.paste(bmpLight, (330,0))
            if (sensor.relative_humidity < 50) and (veml7700.light > 10000):
                os.system('sudo uhubctl -a off -l 2 > /dev/null')
                fan = "off"
                Himage.paste(bmpLight, (330,0))

            with open(fileTime,"a") as file:
                writer = csv.writer(file)
                writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"),sensor.relative_humidity, fah, lux, light, fan])                     
                
            epd.display_Partial(epd.getbuffer(Himage))
            #epd.display_Fast(epd.getbuffer(Himage))
            num = num + 1
            time.sleep(60)
            #if(num == 10):
            # break


    else:
        logging.info("E-paper refreshes quickly")
        # Drawing on the Horizontal image
        epd.init_fast(epd.Seconds_1_5S)
        logging.info("1.Drawing on the Horizontal image...")
        num = 0
        while(True):
            #pull from sensors every loop
            now = datetime.datetime.now()
            cel = sensor.temperature
            fah = (1.8 * cel) + 32
            fah = '%.1f' % fah

            temp = str(fah) + '\xb0' + 'F'
            humidity = '%.1f' % sensor.relative_humidity
            hum = str(humidity) + '%'
            light = '%.f' % veml7700.light
            #light = str(light)
            lux = '%.f' % veml7700.lux
            #lux = str(lux)

            
            Himage = Image.new('1', (epd.width, epd.height), 255)  # 255: clear the frame
            draw = ImageDraw.Draw(Himage)
            draw.text((10, 170), hum, font = font55, fill = 0)
            draw.line((190, 160, 190, 300), fill = 0)
            draw.text((200, 170), temp, font = font55, fill = 0)
            draw.text((0, 10), time.strftime('%I:%M'), font = font85, fill = 0)
            draw.text((215, 65), time.strftime('%p'), font = font24, fille = 0)
            draw.text((0, 95), time.strftime('%A  %D'), font = font35, fille = 0)
            draw.line((0, 150, 400, 150), fill = 0)
            if (sensor.relative_humidity > 50) and (veml7700.light < 10000) :
                os.system('sudo uhubctl -a on -l 2 > /dev/null')
                Himage.paste(bmpFan, (380,10))
                fan = "on"
            if (sensor.relative_humidity < 50) and (veml7700.light < 10000):
                os.system('sudo uhubctl -a off -l 2 > /dev/null')
                fan = "off"
            if (sensor.relative_humidity > 50) and (veml7700.light > 10000):
                os.system('sudo uhubctl -a on -l 2 > /dev/null')
                Himage.paste(bmpFan, (380,10))
                fan = "on"
                Himage.paste(bmpLight, (330,0))
            if (sensor.relative_humidity < 50) and (veml7700.light > 10000):
                os.system('sudo uhubctl -a off -l 2 > /dev/null')
                fan = "off"
                Himage.paste(bmpLight, (330,0))
                
            with open(fileTime,"a") as file:
                writer = csv.writer(file)
                writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"),sensor.relative_humidity, fah, lux, light, fan])      
            epd.display_Partial(epd.getbuffer(Himage))
            #epd.display_Fast(epd.getbuffer(Himage))
            num = num + 1
            time.sleep(60)
            #if(num == 10):
            #    break

    
    logging.info("Clear...")
    epd.init()
    epd.Clear()
    logging.info("Goto Sleep...")
    epd.sleep()
    
except IOError as e:
    logging.info(e)
    
except KeyboardInterrupt:  
    logging.info("ctrl + c:")
    #epd4in2_V2.epdconfig.module_exit(cleanup=True)
    logging.info("init and Clear")
    epd.init()
    epd.Clear()
    GPIO.cleanup()
    print("cleanup done")
    exit()
