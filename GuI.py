import tkinter as tk
# from tkinter import ttk
from tkinter import *
# import tkinter.messagebox
import tkinter.font as tkFont
import os
import numpy as np
import cv2
# import imutils
from PIL import Image
from PIL import ImageTk
import threading
from datetime import datetime
import time
# import subprocess
from detector import DetectorTF2

# ======================================================================
# ================= SETUP THE DETECTION ================================
# ======================================================================


# For preprocessing image
def make_image_square(filename):
    img = cv2.imread(filename)
    # Size of the image
    # s = max(img.shape[0:2])
    s = 640

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    cv2.imwrite(filename, f)

# For preprocessing image
def crop_image():
    for image_index in range(2):
        folder_name = 'data/'
        image_name = folder_name + str(image_index + 1) + '.jpg'
        img = cv2.imread(image_name)

        h, w, c = img.shape

        w_constant = w/3
        h_constant = h/2

        image_part_index = 0

        for index_w in range(3):
            for index_h in range(2):
                start_width = int(w_constant * index_w)
                end_width = int(w_constant * (index_w + 1))

                start_height = int(h_constant * index_h)
                end_height = int(h_constant * (index_h + 1))

                current_index = image_part_index

                # For training image set
                # section_name = 'PL_8_' + str(image_index+1) + '_'
                # file_name = section_name + str(image_index+1) + '_' + str(image_part_index) + '.jpg'

                # For testing image set
                section_name = str(image_index+1) + '/'
                file_name = folder_name + section_name + \
                    str(image_part_index+1) + '.jpg'

                crop_img = img[start_height:end_height, start_width:end_width]

                image_part_index = image_part_index + 1
                cv2.imwrite(file_name, crop_img)

                make_image_square(file_name)

# For detection
def WriteFile(output_dir, file_name, content):
    file_output = os.path.join(output_dir, file_name)
    f = open(file_output, 'a+')
    f.write(content)
    f.close()

# For detection
def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
	total_detected = 0
	timestamp2 = time.time()

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			img = cv2.imread(image_path)
			timestamp1 = time.time()
			det_boxes = detector.DetectFromImage(img)
			elapsed_time = round((time.time() - timestamp1) * 1000) #ms
			img = detector.DisplayDetections(img, det_boxes)

			total_detected = total_detected + len(det_boxes)
			text_to_save = str(file.name) + ':\t' + str(len(det_boxes)) + ' benur detected' + '\t' + '[' + str(elapsed_time/1000) + ' s] \t\n'

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)
				WriteFile(output_dir, 'ResultLog.txt', text_to_save)

	elapsed_time2 = round((time.time() - timestamp2) * 1000) #ms
	final_text_to_save = str(total_detected) + 'benur detected\t' + '[' + str(elapsed_time2/1000) + ' s]'
	if save_output:
		WriteFile(output_dir, 'Final.txt', final_text_to_save)
	return total_detected

# For detection
def execute_tf(model_path, threshold, output_directory, labelmap_path, images_dir, id_list_data = None):
    id_list = id_list_data
    if id_list_data is not None:
        id_list = [int(item) for item in id_list_data.split(',')]

    save_output = True
    if save_output:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # instance of the class DetectorTF2
    detector = DetectorTF2(model_path, labelmap_path,
                            class_id=id_list, threshold=threshold)

    DetectImagesFromFolder(
        detector, images_dir, save_output=True, output_dir=output_directory)


models = ['ssd-mobilenet-v2-5000']
threshold_setup = [0.3]
test_images_folders = ['1', '2']

# For detection
def detect_images():
    detected_total = 0
    total_detection_process = 0
    for threshold in threshold_setup:
        # Generate string for threshold output folder
        threshold_str = str(threshold)
        threshold_str = threshold_str.replace('.', '_')

        for folder in test_images_folders:

            # Generate string for output folder
            folder_subname = folder.replace('/', '_')

            for model in models:
                # Generate output directory
                output_directory = 'output_' + folder_subname + '_' + threshold_str

                detection_model_path = 'models/' + model
                detection_labelmap_path = 'models/shrimp-seed_label_map.pbtxt'
                detection_images_dir = 'data/' + folder
                detection_output_dir = 'data/' + output_directory + '/' + model

                detection_result = execute_tf(detection_model_path, threshold, detection_output_dir, detection_labelmap_path, detection_images_dir)

                detected_total += int(detection_result)
                total_detection_process += 1

    return detected_total / total_detection_process


def start_detection_process():

    #=========== Detection Program here ========#
    crop_image()
    #=========== Should return the result ======#
    detected_result = detect_images()

    return detected_result


# ======================================================================
# ===================== SETUP THE GUI ==================================
# ======================================================================
white       = "#ffffff"
BlackSolid  = "#000000"
font        = "Constantia"
fontButtons = (font, 12)
maxWidth    = 800
maxHeight   = 480

class buttonL:

    def __init__(self, obj, size, position, text,font, fontSize, hoverColor,command=None):
        self.obj= obj
        self.size= size
        self.position= position
        self.font= font
        self.fontSize= fontSize
        self.hoverColor= hoverColor
        self.text= text
        self.command = command
        self.state = True
        self.Button_ = None

    def myfunc(self):
        print("Hello size :" , self.size)
        print("Hello position :" , self.position)
        print("Hello font :" , self.font)
        print("Hello fontSize :" , self.fontSize)
        print("Hello hoverState :" , self.hoverColor)
  
    def changeOnHover(self, obj,colorOnHover, colorOnLeave):
         obj.bind("<Enter>", func=lambda e: obj.config(
             background=colorOnHover))

         obj.bind("<Leave>", func=lambda e: obj.config(
             background=colorOnLeave))
            
    def buttonShow(self):
        fontStyle = tkFont.Font(family= self.font, size=self.fontSize,weight="bold")
        self.Button_ = Button(self.obj,text = self.text, font=fontStyle, width = self.size[0], height = self.size[1],  bg =   self.hoverColor[1] if isinstance(self.hoverColor, list)  == True else  self.hoverColor, compound=TOP,command=self.command)         
        self.Button_.place(x=self.position[0],y=self.position[1])

        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)
    
    def stateButton(self,st):
        self.st=st
        if not self.Button_ == None:
            self.Button_["state"]=self.st

class framecontroller(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
         #Graphics window
        self.mainWindow = self
        self.mainWindow.configure(bg=BlackSolid)
        self.mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
        self.mainWindow.resizable(0,0)
        self.mainWindow.title("SHRICO Satrio Version")
        self.mainWindow.attributes("-fullscreen", True)
        
        # # creating a container
        container = tk.Frame(self.mainWindow) 
        container.configure(bg=BlackSolid)
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        
        self.frames = {} 

        for F in (StartPage,Page1):
  
            frame = F(container, self.mainWindow)

            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
    
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.ratarata=0

        self.configure(bg=BlackSolid)

        fontStyleLabel= tkFont.Font(family="Arial", size=40)
        self.label1 = Label( self, text="Jumlah Benih", bg='#000', fg='#fff', font=fontStyleLabel)
        self.label1.pack()
        self.label1.place(x=80,y=100)

        fontStyleLabel= tkFont.Font(family="Arial", size=100)
        self.label2 = Label(self, text=0, bg='#000', fg='#fff', font=fontStyleLabel)
        self.label2.pack()
        self.label2.place(x=80,y=200)

        fontStyle = tkFont.Font(family= "Arial", size=15,weight="bold")
        self.button1 = buttonL(self,[15,2],[580,40],"Kalibrasi",fontStyle,15,["yellow",white],lambda : [controller.show_frame(Page1)])
        self.button1.buttonShow()

        self.button2 = buttonL(self,[15,2],[580,280],"Hitung Benih",fontStyle,15,["yellow",white],self.Waitcalculate)
        self.button2.buttonShow()

        self.button3 = buttonL(self,[15,2],[580,380],"Matikan Alat",fontStyle,15,["yellow",white],lambda : self.close())
        self.button3.buttonShow()

        

    def Waitcalculate(self):
        fontStyleLabel= tkFont.Font(family="Arial", size=20)
        self.label3 = Label( self, text="Proses Deteksi Sedang Berlangsung", bg='#000', fg='#fff', font=fontStyleLabel)
        self.label3.pack()
        self.label3.place(x=80,y=50)

        self.label2.configure(text="~")

        fontStyleLabel= tkFont.Font(family="Arial", size=15)
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%B %d, %Y %H:%M:%S")
        self.label4 = Label( self, bg='#000', fg='#fff', font=fontStyleLabel)
        self.label4.configure(text="Waktu:\n"+self.dt_string,justify="left")
        self.label4.pack()
        self.label4.place(x=80,y=400)

        self.button1.stateButton("disabled")
        self.button2.stateButton("disabled")
        self.button3.stateButton("disabled")

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.tensorflow)
        self.thread.start()

    def tensorflow(self):
        #================ Process ===================#
        import os
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "tensor.py"
        abs_file_path = os.path.join(script_dir, rel_path)
        # p1=subprocess.check_output('python3 tensor.py', shell=True)
        # value = int(p1.decode('utf-8'))
        value = start_detection_process()
        self.ratarata = value
        #============================================#
        self.stopEvent.set()
        self.Resultcalculate(self.ratarata)

    def Resultcalculate(self,ratarata):
        
        self.label3.configure(text="Proses Deteksi Selesai")
        self.label2.configure(text=ratarata)

        self.button1.stateButton("active")
        self.button1.buttonShow()
        self.button2.stateButton("active")
        self.button2.buttonShow()
        self.button3.stateButton("active")
       
    def close(self):
        import sys
        sys.exit("Close")	



class Page1(tk.Frame):
     
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        self.videoObj = None

        self.configure(bg=BlackSolid)

        fontStyleLabel= tkFont.Font(family="Arial", size=14)
        label1 = Label( self, text="Pastikan Wadah Benih\nUdang Terlihat Jelas\nMelalui Kamera", bg='#000', fg='#fff', font=fontStyleLabel)
        label1.pack()
        label1.place(x=600,y=50)

        fontStyle = tkFont.Font(family= "Arial", size=15,weight="bold")
        button1 = buttonL(self,[12,1],[620,400],"Selesai",fontStyle,15,["yellow",white],lambda : [controller.show_frame(StartPage), videoStream.onClose(self.videoObj)])
        button1.buttonShow()

        button2 = buttonL(self,[12,1],[620,300],"Camera On",fontStyle,15,["yellow",white],lambda : [ videoStream.onStart(self.videoObj)])
        button2.buttonShow()

        self.videoObj = videoStream()



class videoStream(tk.Frame):
    def __init__(self):
    
        self.ret = None
        self.frame = None

        self.thread = None
        self.stopEvent = None 
        self.capWebcam = None

        self.panel = None
    
    def onStart(self):

        self.capWebcam = cv2.VideoCapture(0) 
        if not self.capWebcam.isOpened():
            raise Exception("Could not open video device")
        self.capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop)
        self.thread.start()

    def onClose(self):
        print("[INFO] closing...")
        if not self.panel == None:
            self.panel.destroy()
            self.stopEvent.set()
            self.capWebcam.release()
        

    def videoLoop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                
                self.ret,self.frame = self.capWebcam.read()
           
                if(self.ret==True):
                    image = cv2.flip(self.frame, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)
                
                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                       
                        self.panel = Label(image=image,width=600,height=480)
                        self.panel.image = image
                        self.panel.place(x=0,y=0)
            
                    # otherwise, simply update the panel
                    else:
                     
                        if(not self.panel == None):
                            self.panel.configure(image=image)
                            self.panel.image = image
                else:
               
                    self.panel.destroy()
                    self.panel = None


        except RuntimeError:
            print("[INFO] caught a RuntimeError")

app = framecontroller()
app.mainloop()


