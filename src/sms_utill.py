# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:01:25 2017

@author: Avalanche
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

class sms:
    def __init__(self):
        
        self.subject = 'Monitor test'
        self.from_addr = 'weatherstationsmonitor@gmail.com'
        self.pw = "We4therM0nit0rEmailPa55w0rd"
        self.from_ = 'weather Sstations Monitor'
        
    def log_in_and_send(self, msg):
        
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.ehlo()
            s.starttls()
            s.login(self.from_addr, self.pw)
            s.sendmail(self.from_, msg['To'], msg.as_string() )


        
    def send(self, to='3038184876@vtext.com', text=''):
        
        msg = MIMEText('text')
        msg['From'] = self.from_
        msg['To'] = to
        msg['Subject'] = ''
        
        self.log_in_and_send(msg)
                
    def send_with_attachment(self, to='3038184876@vtext.com', text='', file=''):
        
        if file=='':
            self.send(to=to, text=text)
            return
        attachment = open(file, 'rb').read()
        msg = MIMEMultipart()
        msg['From'] = self.from_
        msg['To'] = to
        msg['Subject'] = ''
        msg_text = MIMEText(text)
        msg.attach(msg_text)
        image = MIMEImage(attachment, name=os.path.basename(file))
        msg.attach(image)
        self.log_in_and_send(msg)
        
if __name__ == '__main__':
    
    message = sms()
    contacts = {'Ron':'3038184876@vtext.com', 'Ryan': '7204173717@vtext.com'}
#    message.send_with_attachment(to = contacts['Ron'], text='Sending image...', file=r'C:\Users\Avalanche\Pictures\smile.jpg')
    message.send(text='Test from Ron')