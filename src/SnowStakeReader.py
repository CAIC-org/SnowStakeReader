import threading
#from multiprocessing import Pool
import time
import tkinter as Tk
import json
import cv2
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import logging
import myutill
from sms_utill import sms


class SnowDepthReader(object):

    def __init__(self, camera):

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] (%(threadName)-9s) %(message)s',)
        self.log = logging.getLogger(__name__)
        self.drawing = False
        self.time = 0
        self.image_file = ''
        self.img = np.empty(0)
        self.imgt = np.empty(0)
        self.hh = 0
        self.hl = 0
        self.wh = 0
        self.wl = 0
        self.ix = 0
        self.iy = 0
        for n, c in camera.items():
            self.name = n
            self.camera = c
            self.rotation = c['rotation']
        with open('notification_list.json', 'r') as f:
            self.notification_list = json.load(f)

        self.read_snow_stake()

    def download_new_image(self):
        
        def local_address(url):
            local = url.find('http') < 0
            return local

        if local_address(str(self.camera['url'])):
            img_list = glob.glob(str(self.camera['url']+'*.jpg'))
            last_img = max(img_list, key=os.path.getctime)
            self.img = cv2.imread(last_img)
        else:
            self.img = myutill.download_last_image(str(self.camera['url']))
        
        return

    # this functions rotate the image so the snow stake will be vertical and convert to a threshold image.
    def arrange_img(self):

        rows, cols = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.camera['rotation'], 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        self.img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        img = cv2.GaussianBlur(img, (7, 7), 0)

    # This function return the roi (where is the snow stake in the image)
    def get_roi(self):

        # read the rois from the station
        if 'roi' in self.camera:
            return self.camera['roi']

    def read_snow_stake(self):

        """

        :rtype:
        """
        self.download_new_image()
        self.arrange_img()
        roi_bound = self.get_roi()
        roi = self.img[int(roi_bound['hl']):int(roi_bound['hh']), int(roi_bound['wl']):int(roi_bound['wh'])]
        try: # Mostly white stick
            snow_line = np.where(roi<roi.mean()-roi.std())[0][-1]
        except IndexError: # Mostly dark stick
            snow_line = np.where(roi<roi.mean()+roi.std())[0][-1]
        rows, cols = roi.shape
        depth = self.camera['max depth'] * (rows-snow_line) / rows
        #Round depth to the nearest 1/2 inch:
        depth = round(depth * 2)/2
        if os.path.isfile(self.camera['output file']):
            old_sh = pd.read_csv(self.camera['output file'])
        else:
            old_sh = pd.DataFrame(columns=['Time', 'HS'])
        new_sh = pd.DataFrame([{'Time': str(datetime.now())[:-10], 'HS': depth}])
        sh = pd.concat([new_sh, old_sh], ignore_index=True)[['Time', 'HS']]
        sh.to_csv(self.camera['output file'], index=False)        
        depths = sh.iloc[:48, :][::-1]
        fig, _ = plt.subplots(figsize=(12, 7))
        depths.set_index('Time').plot(grid=True, rot=30)
        plt.title('{0} snow depth'.format(self.name))
        plt.ylim(0, depths['HS'].max()*1.5)#self.camera['max depth']+2)
        plt.ylabel('Snow depth in in.')
        plt.tight_layout()
        plt.savefig('../data/{0}/{0}_snow_depth.png'.format(self.name))
        if key in self.notification_list:
            if (time.time()-self.notification_list[key]['delay'] > 6*3600):
                if key == 'Loveland':
                    if depth > 4 and (time.time()-self.notification_list[key]['delay'] > 6*3600):
                        message = sms()
                        if datetime.now().hour > 16 or datetime.now().hour < 7:
                            for name, number in self.notification_list[key].items():
                                if name == 'delay':
                                    continue
                                message.send(to=number, text='Hi {0}, The {1} snow stake is at {2}'.format(name, key, depth))
                            self.notification_list[key]['delay'] = time.time()
                            with open('notification_list.json', 'w') as f:
                                json.dump(self.notification_list, f)
                if key == 'Monument':
                    if depth - sh.iloc[-24, 1] > 6:                        
                        message = sms()
                        for name, number in self.notification_list[key].items():
                            if name == 'delay':
                                continue
                            message.send(to=number, text='Hi {0}, The {1} snow stake is at {2}'.format(name, key, depth))
                        self.notification_list[key]['delay'] = time.time()
                        with open('notification_list.json', 'w') as f:
                            json.dump(self.notification_list, f)
                    
'''
Callback function for image angle setting slide bar
'''
def on_change():

    pass
        

class NewCamera(object):

    def __init__(self, url=''):

        self.drawing = False
        self.hh = 0
        self.hl = 0
        self.wh = 0
        self.wl = 0
        self.ix = 0
        self.iy = 0
        self.rotation = 0
        self.zoom = 1
        self.name = ''
        self.camera = {}
        self.cam = {}
        self.root = Tk.Tk()
        self.root.wm_title('Enter new camera')

        self.label1 = Tk.Label(self.root, text='Name:')
        self.E1 = Tk.Entry(self.root, bd=5)

        defalult_url = Tk.StringVar(self.root, url)
        self.label2 = Tk.Label(self.root, text='URL:')
        self.E2 = Tk.Entry(self.root, textextvariable=defalult_url, bd=5)

        self.label3 = Tk.Label(self.root, text='Max height:')
        self.E3 = Tk.Entry(self.root, bd=5)

        self.submit = Tk.Button(self.root, text="Submit", command=self.save_new_camera)

        self.label1.pack()
        self.E1.pack()
        self.label2.pack()
        self.E2.pack()
        self.label3.pack()
        self.E3.pack()
        self.submit.pack(side=Tk.BOTTOM)

        self.root.mainloop()

    def download_image(self):
        
        def local_address(url):    
            return url.find('http') != 0
        
        url = self.cam['url']
        if local_address(str(self.cam['url'])):
            if os.path.splitext(self.cam['url'])[1] not in ['.jpg', '.png']:
                img_list = glob.glob(str(self.cam['url']+'*.jpg'))
                url = max(img_list, key=os.path.getctime)
            self.img = cv2.imread(url)
        else:
            self.img = myutill.download_image(self.cam['url'])
        return self.img
    

    def draw_roi(self, event, x, y, flags, param):

        if self.drawing:
            self.imgt = cv2.resize(self.img.copy(), None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_AREA)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix = x
            self.iy = y

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                print(x, y)
                self.imgt = cv2.rectangle(self.imgt, (self.ix, self.iy), (x, y), (0, 0, 255), 2)

        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.imgt, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
            self.wh = max(self.ix, x)/self.zoom
            self.wl = min(self.ix, x)/self.zoom
            self.hh = max(self.iy, y)/self.zoom
            self.hl = min(self.iy, y)/self.zoom

    def set_roi(self):

        """

        :rtype: object
        """
        self.download_image()

        cv2.namedWindow('set camera images')
        cv2.createTrackbar('img_rotation', 'set camera images', 0, 8, on_change)
        cv2.setMouseCallback('set camera images', self.draw_roi)
        self.imgt = self.img.copy()
        rows, cols = self.img.shape[:2]
        rotate = 0
        roi = {}
        # Set the image rotation
        while True:
            cv2.imshow('set camera images', self.imgt)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            rotate = (cv2.getTrackbarPos('img_rotation', 'set camera images') - 4)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate, 1)
            self.imgt = cv2.warpAffine(self.img, M, (cols, rows))
            
        # Set anchore
        self.imf = self.imgt.copy()
        self.zoom = 1
        while True:
            cv2.imshow('set camera images', self.imgt)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            if k == ord('o'):
                self.zoom = self.zoom/1.5
                self.imgt = cv2.resize(self.img.copy(), None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_AREA)
            anchore_ul_x, anchore_ul_y = self.wl, self.hl
            cv2.imwrite(self.name + '_anc.jpg', self.imgt[self.hl:self.hh, self.wl:self.wh,:])
            anc_h, anc_w = self.hh-self.hl, self.wh-self.wl
                        
        # Set ROI
        self.img = self.imgt.copy()
        self.zoom = 1
        while True:
            cv2.imshow('set camera images', self.imgt)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
            if k == ord('o'):
                self.zoom = self.zoom/1.5
                self.imgt = cv2.resize(self.img.copy(), None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_AREA)
            roi = {'upper_left_y': self.hh, 'hl': self.hl, 'wh': self.wh, 'wl': self.wl}

        cv2.destroyAllWindows()
        self.cam['rotation'] = rotate
        self.cam['anchore'] = self.name + '_anc.jpg'
        self.cam['roi'] = roi

    def save_new_camera(self):

        if os.path.isfile('cameras.json'):
            with open('cameras.json', 'r') as f:
                cameras = json.load(f)
        else:
            cameras = {}
        url = self.E2.get()
        self.cam['url'] = url#[:url.rfind('/')+1]
        self.cam['max depth'] = float(self.E3.get())
        self.cam['ext'] = 'jpg'
        self.name = self.E1.get()
        self.cam['output file'] = os.path.join('../data/{0}/{0}.dat'.format(self.name))
        cameras[self.name] = self.cam
        self.root.destroy()
        self.set_roi()
        with open('cameras.json', 'w') as f:
            json.dump(cameras, f)
        return
    
    def get_new_camera(self):

        return self.camera


# A worker thread class that listen to new camera entry
class NewCameraEntryListener(object):

    def __init__(self, interval=1, cameras = {}):
        """ Constructor

        :type interval: int, list of cameras
        :param interval: Check interval, in seconds, cameras: dic with the cameras to add the new one
        """
        self.interval = interval
        self.cameras = cameras

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):

        while True:
            time.sleep(self.interval)
            enc = input('\nClick n and Enter to add a new camera \n')
            if enc == 'n':
                new_cam = NewCamera().get_new_camera()
                for name in new_cam:
                    self.cameras[name] = new_cam[name]
                    with open('cameras.json', 'w') as fp:
                        json.dump(cameras, fp)


def send_to_snow_readers(camera):
    

    _ = SnowDepthReader(camera)



if __name__ =='__main__':
    
    cam = NewCamera()

    '''
    '''
    

#cam = NewCamera()
#while True:
    with open('cameras.json', 'r') as fp:
        cameras = json.load(fp)
    for key, cam in cameras.items():
        camera = key[0].upper() + key[1:].lower()
        os.makedirs('../data/{}'.format(camera), exist_ok=True)
        #os.makedirs('/home/www/html/snow_stake_reader/data/{0}/'.format(camera), exist_ok=True)
        reader = SnowDepthReader({key: cam})
        
#    cameras_list = [{name: cam} for name, cam in cameras.items()]
#    print(cameras_list)
#    pool = Pool(4)
#    pool.map_async(SnowDepthReader, cameras_list)
#    pool.close()
#    pool.join()
#
#    time.sleep(120) # wait 15 min between reads





'''
read_dispatcher = ReadDispatcher()
read_dispatcher.daemon = True
read_dispatcher.start()

# enter new camera
while True:
  time.sleep(1)
  i = raw_input('\nPress n + Enter to enter a new camera\n')
  if i == 'n':
    with open('cameras.json', 'r') as fp:
          cameras = json.load(fp)

    new_cam = NewCamera().get_new_camera()
    for name in new_cam:
        cameras[name] = new_cam[name]

    with open('cameras.json', 'w') as fp:
        json.dump(cameras, fp)
'''
