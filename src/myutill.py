from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import numpy as np
import cv2
import os
#from twilio.rest import Client

def page_is_image(url):
    
    image_types = ['.jpg', '.png']
    for i in image_types:
        if len(url)-(url.find(i)+len(i))==0:
            return True
        return False

def download_image(url, return_as='opencv mat'):

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    page = urlopen(req).read()
    if page_is_image(url):
        img = page
    else:
        soup = BeautifulSoup(page, 'html.parser')
        image = soup.find('img')
        image_url = os.path.join(url, image['src'])
        img = urlopen(image_url).read()
    if return_as=='opencv mat':
        np_img = np.array(bytearray(img), dtype=np.uint8)
        img = cv2.imdecode(np_img, -1)
    return img


def download_last_image(url, return_as='opencv mat'):

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    if not page_is_image(url):
        page = urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')
        links = []
        for image in soup.findAll('img'):
            links.append(os.path.join(url, image['src']))
        img = urlopen(links[-1]).read()
    else:
        img = urlopen(req).read()
    if return_as=='opencv mat':
        np_img = np.array(bytearray(img), dtype=np.uint8)
        img = cv2.imdecode(np_img, -1)
    return img

#This function receives a phone number and a message and send an SMS

def send_sms(name='', to_number='', text_message=''):
    
    if to_number == '' or text_message == '':
        return
    
    # Clean the recipient phone number 
    for i in '~`!@#$%^&*()_-=+<>.,:;/?|{}[]':
        to_number = to_number.replace(i, '')
    to_number  = '+1' + to_number
    auth_token  = '40b18ffcc6bce1bd3e13ed98126f305c'
    account_sid = 'AC456b6f49e9dd7b3566fb39cdcaaf7651'
    
    client = Client(account_sid, auth_token)
    message = client.messages.create(to=to_number, 
                                     from_='17208973957', 
                                     body=text_message)


if __name__=='__main__':
    img = download_image('https://assets2.webcam.io/w/RzJLN9/latest.jpg')
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()