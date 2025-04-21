#import os as _os
#_os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\bin")
import tkinter as tk
# from tkinter import Toplevel
from tkinter import *
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from datetime import datetime
# from tensorflow.lite.python.interpreter import Interpreter ,load_delegate
# import tensorflow as tf
# import tensorflow as tf
from threading import Thread , Lock
import os
import numpy as np
from random import randint
from ultralytics import YOLO
# cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin"
import pandas as pd
from tkinter import filedialog

# os.environ["PATH"] += os.pathsep + cuda_path
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.cap = None
        self.video_writer = None
        self.detection_thread = None
        self.detection_lock = Lock()
        self.detection_results = []
        self.labels = ['Bisolvon', 'BowDaeng', 'Carbon', 'Histatab', 'Jiwa Herb', 'MOOKFY', 'MYPARA', 'NacLong', 'Nasolin', 'Sara', 'Solmax', 'Tylenol']

        self.model = YOLO("best.pt")
        self.image_icon = tk.PhotoImage(file="icon\\1159798.png")

        pil_image = Image.open("image\\Bisolvon.jpg")
        self.bisolvon = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\BowDaeng.jpg")
        self.bowdaeng = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Carbon.jpg")
        self.carbon = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Histatab.jpg")
        self.histatab = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Jiwa Herb.jpg")
        self.jiwaherb = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\MOOKFY.jpg")
        self.mookfy = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\MYPARA.jpg")
        self.mypara = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\NacLong.jpg")
        self.naclong = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Nasolin.jpg")
        self.nasolin = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Sara.jpg")
        self.sara = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Solmax.jpg")
        self.solmax = ImageTk.PhotoImage(pil_image)

        pil_image = Image.open("image\\Tylenol.jpg")
        self.tylenol = ImageTk.PhotoImage(pil_image)

        self.label = tk.Label(window, image=self.image_icon, width= 1200, height=720)
        self.label.pack(expand=False, fill="both",anchor="n")

        self.label_clock = tk.Label(window, text="", font=("Arial", 20))
        self.label_clock.place(x=10, y=10)
        
        self.upload_image = tk.Button(window,text="Upload Image",command=self.upload_image_action,width=10,height=5)
        self.upload_image.place(x=1050,y=700)

        self.btn_capture = tk.Button(window,text="Capture",command=self.capture_screenshot,width=10,height=5)
        self.btn_capture.place(x=1150,y=700)

        self.btn_toggle_camera = tk.Button(window, text="Camera", command=self.toggle_camera, width=10, height=5)
        self.btn_toggle_camera.place(x=1250, y=700)
        
        self.btn_toggle_detecting = tk.Button(window, text="Detect", command=self.toggle_detecting, width=10, height=5)
        self.btn_toggle_detecting.place(x=1350, y=700)
        
        self.detecting_enabled = False
        
        
        self.label1 = tk.Label(text="Camera: Off", font=("Arial", 20))
        self.label1.place(x=640, y=700)

        self.label2 = tk.Label(text="Detection: Off", font=("Arial", 20))
        self.label2.place(x=640, y=750)
        
        self.update()
        self.update_clock()
        self.window.mainloop()
    def upload_image_action(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        
        if file_path:
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

            results2 = self.model.predict(img_rgb,conf=0.1,verbose=False)
            boxes = results2[0].boxes.xyxy.tolist()
            classes = results2[0].boxes.cls.tolist()
            names = results2[0].names
            confidences = results2[0].boxes.conf.tolist()
            detection_data2 = []
            for box , cls , conf in zip(boxes,classes,confidences):
                if conf > 0.5:
                    xmin , ymin , xmax , ymax = map(int,box)
                    cv2.rectangle(img_rgb, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    label1 = f"{names[int(cls)]} : {conf}"
                    cv2.putText(img_rgb, label1,
                                (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    detection_data2.append((names[int(cls)], conf))
            timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename2 = f"screenshot_{timestamp2}.jpg"
            cv2.imwrite(filename2,cv2.cvtColor(img_rgb , cv2.COLOR_RGB2BGR))
            self.show_popup(filename2,detection_data2)

    def update_clock(self):
        str_current_time = datetime.now().strftime("%H:%M:%S")
        self.label_clock.config(text=str_current_time)
        self.window.after(1000, self.update_clock)

    def toggle_camera(self):
        if self.cap is None:
            self.frame_rate_calc = 1
            self.freq = cv2.getTickFrequency()
            self.cap = VideoStream(resolution=(1290,670),framerate=30).start()
            self.frame_width = 1290  # Set your desired width
            self.frame_height = 670  # Set your desired height
            # self.btn_toggle_camera.config(image=self.notcam_icon, width=85, height=68)
            self.label1.config(text="Camera: On")

        else:
            self.cap.stop()
            self.cap = None
            self.label.config(image=self.image_icon, width= 1200, height=720)
            # self.btn_toggle_camera.config(image=self.cam_icon, width=85, height=68)
            self.label1.config(text="Camera: Off")
            


    def toggle_detecting(self):
        self.detecting_enabled = not self.detecting_enabled # เปลี่ยนสถานะการตรวจจับวัตถุ
        if self.detecting_enabled:
            # self.btn_toggle_detecting.config(image=self.notdetect_icon, width=85, height=68) # เปลี่ยนข้อความปุ่มเป็น "ปิดตรวจจับ"
            self.label2.config(text="Detection: On")
        else:
            # self.btn_toggle_detecting.config(image=self.detect_icon, width=85, height=68) # เปลี่ยนข้อความปุ่มเป็น "เปิดตรวจจับ"
            self.label2.config(text="Detection: Off")

    def update(self):
        if self.cap is not None: # ตรวจสอบว่ากล้องถูกเปิดหรือไม่
            t1 = cv2.getTickCount()
            frame = self.cap.read()
            imH, imW, _ = frame.shape
            if self.detecting_enabled: # ตรวจสอบว่าต้องการทำการตรวจจับหรือไม่
                self.process_detection(frame,imH,imW)
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.config(image=photo)
            self.label.image = photo
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            self.frame_rate_calc = 1/time1
            if self.video_writer is not None: # ตรวจสอบว่ากล้องถูกเปิดหรือไม่
                self.video_writer.write(frame)
        self.window.after(15, self.update)
    def process_detection(self,frame,imH,imW):
        results = self.model.predict(frame,conf=0.1,verbose=False)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()
        detection_data = []
        for box , cls , conf in zip(boxes,classes,confidences):
            if conf > 0.5:
                xmin , ymin , xmax , ymax = map(int,box)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                label1 = f"{names[int(cls)]} : {conf}"
                cv2.putText(frame, label1,
                            (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                detection_data.append((names[int(cls)], conf))
        if detection_data:
            self.last_detection = (frame.copy() ,detection_data)
    def capture_screenshot(self):
        if self.cap is not None and self.last_detection:
            frame , detection_data = self.last_detection
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename,cv2.cvtColor(frame , cv2.COLOR_RGB2BGR))
            self.show_popup(filename,detection_data)
    def show_popup(self,image_path,detection_data):
        popup = Toplevel(self.window)
        popup.title("Captured image")
        image_1 = Image.open(image_path)
        image_1 = image_1.resize((600,400))
        photo_1 = ImageTk.PhotoImage(image_1)
        
        label_1 = tk.Label(popup,image=photo_1)
        label_1.image = photo_1
        label_1.pack()
        info_text = "Detected Objects:\n"
        for cls, conf in detection_data:
            info_text += f"- {cls} ({conf:.2f})\n"

        label_2 = tk.Label(popup, text=info_text, font=("Arial", 12))
        label_2.pack()
        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)

        def show_dropdown(cls,conf):
            # dropdown_menu = Menu(popup,tearoff=0)
            # dropdown_menu.add_command(label=f"Class : {cls}")
            # dropdown_menu.add_command(label=f"Confidence : {conf:.2f}")
            popup2 = Toplevel(self.window)
            popup2.title("Information Description")
            if cls=="Tylenol":
                label_4 = tk.Label(popup2, image=self.tylenol,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาบรรเทาปวด ลดไข้\n"
                info_text_2+="สรรพคุณ : ใช้บรรเทาอาการปวดในระดับอ่อนถึงปานกลาง ปวดศรีษะทั่วไป ปวดกล้ามเนื้อ ลดไข้เนื่องจากหวัด\n"
                info_text_2+="การใช้งาน:✔️รับประทานครั้งละ 1-2 เม็ด  วันละ 3-4 ครั้ง สำหรับยาเม็ดขนาด 500 มิลลิกรัม  \n✔️ผู้ใช้มีน้ำหนักตัว 34-50 กิโลกรัม ให้รับประทานยาครั้งละ 1 เม็ด\n✔️ น้ำหนักตัว 51-67 กิโลกรัม ให้รับประทานยาครั้งละ 1 เม็ดครี่ง และถ้ามีน้ำหนักตัว 68 กิโลกรัมขึ้นไป ให้รับประทานยาครั้งละ 2 เม็ด\n"
            elif cls =="Sara":
                label_4 = tk.Label(popup2, image=self.sara,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาบรรเทาปวด ลดไข้\n"
                info_text_2+="สรรพคุณ:ใช้บรรเทาอาการปวดระดับอ่อนถึงปานกลาง \nเช่น ปวดศีรษะ ปวดกล้ามเนื้อ และช่วยลดไข้จากอาการหวัด\n"
                info_text_2+="การใช้งาน:✔️ รับประทานครั้งละ 1-2 เม็ด วันละ 3-4 ครั้ง (สำหรับยาเม็ดขนาด 500 มิลลิกรัม)\n✔️ ผู้ที่มีน้ำหนักตัว 34-50 กิโลกรัม รับประทานครั้งละ 1 เม็ด\n✔️ ผู้ที่มีน้ำหนักตัว 51-67 กิโลกรัม รับประทานครั้งละ 1.5 เม็ด\n✔️ ผู้ที่มีน้ำหนักตัว 68 กิโลกรัมขึ้นไป รับประทานครั้งละ 2 เม็ด\n"
            elif cls=="MYPARA":
                label_4 = tk.Label(popup2, image=self.mypara,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาบรรเทาปวด ลดไข้\n"
                info_text_2+="สรรพคุณ:ใช้บรรเทาอาการปวดระดับอ่อนถึงปานกลาง \nเช่น ปวดศีรษะ ปวดกล้ามเนื้อ และช่วยลดไข้จากอาการหวัด\n"
                info_text_2+="การใช้งาน:✔️ รับประทานครั้งละ 1-2 เม็ด วันละ 3-4 ครั้ง (สำหรับยาเม็ดขนาด 500 มิลลิกรัม)\n✔️ ผู้ที่มีน้ำหนักตัว 34-50 กิโลกรัม รับประทานครั้งละ 1 เม็ด\n✔️ ผู้ที่มีน้ำหนักตัว 51-67 กิโลกรัม รับประทานครั้งละ 1.5 เม็ด\n✔️ ผู้ที่มีน้ำหนักตัว 68 กิโลกรัมขึ้นไป รับประทานครั้งละ 2 เม็ด\n"
            elif cls=="BowDaeng":
                label_4 = tk.Label(popup2, image=self.bowdaeng,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ท้องเสีย\n"
                info_text_2+="สรรพคุณ:ช่วยบรรเทาอาการท้องเสียเฉียบพลัน ลดอาการปวดเกร็งในช่องท้อง และช่วยปรับสมดุลระบบทางเดินอาหาร\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 2 เม็ด วันละ 3-4 ครั้ง หลังอาหารหรือเมื่อมีอาการ\n✔️ เด็กอายุ 6-12 ปี รับประทานครั้งละ 1 เม็ด วันละ 3-4 ครั้ง\n✔️ ดื่มน้ำตามมาก ๆ และหลีกเลี่ยงอาหารที่กระตุ้นอาการท้องเสีย\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ในผู้ที่มีไข้สูงหรือสงสัยว่ามีการติดเชื้อแบคทีเรียรุนแรง\n❌ หากอาการไม่ดีขึ้นภายใน 2 วัน ควรปรึกษาแพทย์\n"
            elif cls=="Jiwa Herb":
                label_4 = tk.Label(popup2, image=self.jiwaherb,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ท้องเสีย\n"
                info_text_2+="สรรพคุณ:ช่วยบรรเทาอาการแน่นท้อง ท้องอืด ท้องเฟ้อ ขับลม และช่วยปรับสมดุลระบบทางเดินอาหาร\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1-2 แคปซูล วันละ 3 ครั้ง หลังอาหาร\n✔️ สามารถรับประทานต่อเนื่องได้ตามความจำเป็น หรือจนกว่าอาการจะดีขึ้น\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ในสตรีมีครรภ์และให้นมบุตร เว้นแต่ได้รับคำแนะนำจากแพทย์\n❌ หากมีอาการแพ้ หรืออาการไม่ดีขึ้นภายใน 3 วัน ควรหยุดใช้และปรึกษาแพทย์\n"
            elif cls=="Carbon":
                label_4 = tk.Label(popup2, image=self.carbon,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ท้องเสีย\n"
                info_text_2+="สรรพคุณ:✔️ ช่วยดูดซับสารพิษในทางเดินอาหาร\n✔️ ลดอาการท้องอืด ท้องเฟ้อจากแก๊สในกระเพาะอาหาร\n✔️ ใช้เป็นตัวช่วยในกรณีอาหารเป็นพิษ หรือรับสารที่อาจเป็นอันตรายเข้าสู่ร่างกาย\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 2-4 เม็ด วันละ 3-4 ครั้ง หรือตามแพทย์แนะนำ\n✔️ เด็ก รับประทานครั้งละ 1-2 เม็ด วันละ 3-4 ครั้ง หรือตามแพทย์แนะนำ\n✔️ ดื่มน้ำมาก ๆ หลังรับประทานเพื่อช่วยในการทำงานของถ่านดูดซับ\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ร่วมกับยาอื่น เนื่องจากอาจลดประสิทธิภาพของยา ควรรับประทานห่างกันอย่างน้อย 2 ชั่วโมง\n❌ ไม่ควรใช้ติดต่อกันเป็นเวลานาน หากอาการไม่ดีขึ้น ควรปรึกษาแพทย์\n"
            elif cls=="Nasolin":
                label_4 = tk.Label(popup2, image=self.nasolin,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้แพ้ ลดน้ำมูก\n"
                info_text_2+="สรรพคุณ:ซึ่งเป็นยาในกลุ่มยาแก้คัดจมูกที่ใช้สำหรับบรรเทาอาการคัดจมูกจากหวัดหรือโรคภูมิแพ้ \nโดยการทำงานของมันจะช่วยให้หลอดเลือดในจมูกหดตัวและลดการอักเสบ ทำให้จมูกโล่งขึ้น\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 2-4 เม็ด วันละ 3-4 ครั้ง หรือตามแพทย์แนะนำ\n✔️ เด็ก รับประทานครั้งละ 1-2 เม็ด วันละ 3-4 ครั้ง หรือตามแพทย์แนะนำ\n✔️ ดื่มน้ำมาก ๆ หลังรับประทานเพื่อช่วยในการทำงานของถ่านดูดซับ\n"
                info_text_2+="ข้อควรระวัง:❌ ห้ามใช้ยานี้ติดต่อกันเกิน 3-5 วัน เนื่องจากการใช้ยานานเกินไปอาจทำให้เกิดภาวะ \"rebound congestion\" หรือการคัดจมูกกลับมาอีกเมื่อหยุดใช้ยา\nโรคประจำตัว: ผู้ที่มีปัญหาสุขภาพ เช่น โรคหัวใจ ความดันโลหิตสูง (Hypertension) หรือโรคหลอดเลือด ควรปรึกษาแพทย์ก่อนใช้ เพราะยาอาจมีผลต่อหลอดเลือดการตั้งครรภ์และให้นมบุตร: การใช้ยาในช่วงตั้งครรภ์หรือให้นมบุตรควรได้รับคำแนะนำจากแพทย์\nอาการข้างเคียง: อาจเกิดอาการระคายเคืองที่จมูก เช่น แสบหรือคัน หรือหากใช้มากเกินไปอาจทำให้เกิดการบวมของเยื่อบุจมูกการใช้ร่วมกับยาอื่น: \nควรระมัดระวังการใช้ร่วมกับยาบางชนิด เช่น ยาที่มีผลต่อระบบประสาทหรือยาอื่นๆ ที่อาจมีปฏิกิริยากับยานี้หากมีอาการข้างเคียงที่รุนแรงหรือข้อสงสัยเกี่ยวกับการใช้ยานี้ ควรปรึกษาแพทย์หรือเภสัชกรเพื่อคำแนะนำที่เหมาะสม\n"
            elif cls=="Bisolvon":
                label_4 = tk.Label(popup2, image=self.bisolvon,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ไอ ขับเสมหะ\n"
                info_text_2+="สรรพคุณ:✔️ ช่วยละลายเสมหะ ทำให้เสมหะเหลวและขับออกง่ายขึ้น\n✔️ บรรเทาอาการไอที่มีเสมหะเหนียวข้น\n✔️ เหมาะสำหรับผู้ที่มีอาการไอจากไข้หวัด หลอดลมอักเสบ หรือโรคระบบทางเดินหายใจ\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1 เม็ด วันละ 3 ครั้ง หลังอาหาร\n✔️ เด็กอายุ 6-12 ปี รับประทานครั้งละ ½ เม็ด วันละ 3 ครั้ง\n✔️ ควรดื่มน้ำตามมาก ๆ เพื่อช่วยละลายเสมหะและขับออกได้ง่ายขึ้น\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ในผู้ที่แพ้ยา Bromhexine หรือส่วนประกอบของยา\n❌ หากมีอาการแพ้ เช่น ผื่นคัน หรือหายใจลำบาก ควรหยุดใช้และรีบพบแพทย์\n❌ หากอาการไอไม่ดีขึ้นภายใน 7 วัน ควรปรึกษาแพทย์\n"
            elif cls=="Histatab":
                label_4 = tk.Label(popup2, image=self.histatab,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้แพ้ ลดน้ำมูก\n"
                info_text_2+="สรรพคุณ:✔️ บรรเทาอาการแพ้ เช่น คัดจมูก น้ำมูกไหล จาม คันตา หรือคันจมูก\n✔️ ใช้รักษาอาการลมพิษ และอาการคันจากสาเหตุต่าง ๆ\n✔️ ลดอาการแพ้จากแมลงกัดต่อย หรือแพ้ฝุ่นละออง\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1 เม็ด วันละ 1-3 ครั้ง ตามแพทย์แนะนำ\n✔️ เด็กอายุ 6-12 ปี รับประทานครั้งละ ½ เม็ด วันละ 1-3 ครั้ง ตามแพทย์แนะนำ\n✔️ ควรรับประทานพร้อมน้ำเปล่า และสามารถรับประทานก่อนหรือหลังอาหารก็ได้\n"
                info_text_2+="ข้อควรระวัง:❌ อาจทำให้ง่วงนอนได้ ควรหลีกเลี่ยงการขับรถหรือใช้เครื่องจักรหลังรับประทานยา\n❌ หลีกเลี่ยงการใช้ร่วมกับแอลกอฮอล์ เพราะอาจเสริมฤทธิ์ง่วงนอน\n❌ หากมีอาการแพ้ เช่น ผื่นขึ้น หรือหายใจลำบาก ควรหยุดใช้และรีบพบแพทย์\n"
            elif cls=="MOOKFY":
                label_4 = tk.Label(popup2, image=self.mookfy,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้แพ้ ลดน้ำมูก\n"
                info_text_2+="สรรพคุณ:✔️ ช่วยบรรเทาอาการไอ ขับเสมหะ และลดอาการระคายเคืองในลำคอ\n✔️ ช่วยให้เสมหะหลุดออกได้ง่ายขึ้น ทำให้หายใจสะดวกขึ้น\n✔️ เหมาะสำหรับผู้ที่มีเสมหะข้นเหนียว หรือมีอาการไอเรื้อรัง\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1-2 เม็ด วันละ 3 ครั้ง หลังอาหาร\n✔️ เด็กอายุ 6-12 ปี รับประทานครั้งละ 1 เม็ด วันละ 3 ครั้ง หลังอาหาร\n✔️ ควรดื่มน้ำตามมาก ๆ เพื่อช่วยให้เสมหะขับออกง่ายขึ้น\n"
                info_text_2+="ข้อควรระวัง:❌ อาจทำให้ง่วงนอนได้ ควรหลีกเลี่ยงการขับรถหรือใช้เครื่องจักรหลังรับประทานยา\n❌ หลีกเลี่ยงการใช้ร่วมกับแอลกอฮอล์ เพราะอาจเสริมฤทธิ์ง่วงนอน\n❌ หากมีอาการแพ้ เช่น ผื่นขึ้น หรือหายใจลำบาก ควรหยุดใช้และรีบพบแพทย์\n"
            elif cls=="NacLong":
                label_4 = tk.Label(popup2, image=self.naclong,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ไอ ขับเสมหะ\n"
                info_text_2+="สรรพคุณ:✔️ ช่วยละลายเสมหะ ขับเสมหะออกจากทางเดินหายใจ\n✔️ ใช้รักษาโรคเกี่ยวกับระบบทางเดินหายใจที่มีเสมหะมาก เช่น หลอดลมอักเสบเรื้อรัง โรคปอดอุดกั้นเรื้อรัง (COPD)\n✔️ มีฤทธิ์ช่วยต้านอนุมูลอิสระ และปกป้องเซลล์จากความเสียหาย\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1 เม็ด (600 มก.) วันละ 1-2 ครั้ง ตามแพทย์แนะนำ\n✔️ เด็กอายุต่ำกว่า 12 ปี ควรใช้ตามคำแนะนำของแพทย์\n✔️ ควรดื่มน้ำมาก ๆ เพื่อช่วยให้เสมหะถูกขับออกง่ายขึ้น\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ในผู้ที่แพ้ยา Acetylcysteine หรือส่วนประกอบของยา\n❌ หากมีอาการแพ้ เช่น ผื่นขึ้น แน่นหน้าอก หายใจลำบาก ควรหยุดใช้และรีบพบแพทย์\n❌ หลีกเลี่ยงการใช้ร่วมกับยาแก้ไอที่มีฤทธิ์กดอาการไอ เพราะอาจทำให้เสมหะคั่งค้างในปอด\n"
            elif cls=="Solmax":
                label_4 = tk.Label(popup2, image=self.solmax,width= 300, height=300)
                label_4.pack()
                info_text_2 = f"Class : {cls}\nConfidence : {conf:.2f}\n"
                info_text_2+="กลุ่มยาแก้ไอ ขับเสมหะ\n"
                info_text_2+="สรรพคุณ:✔️ ใช้บรรเทาอาการไอ ละลายเสมหะ และช่วยให้เสมหะขับออกง่ายขึ้น\n✔️ เหมาะสำหรับผู้ที่มีอาการไอจากไข้หวัด หรือโรคทางเดินหายใจที่มีเสมหะมาก\n"
                info_text_2+="การใช้งาน:✔️ ผู้ใหญ่ รับประทานครั้งละ 1 เม็ด วันละ 2-3 ครั้ง หลังอาหาร\n✔️ เด็กอายุ 6-12 ปี รับประทานครั้งละ ½ เม็ด วันละ 2-3 ครั้ง ตามแพทย์แนะนำ\n✔️ ควรดื่มน้ำมาก ๆ เพื่อช่วยให้เสมหะถูกขับออกง่ายขึ้น\n"
                info_text_2+="ข้อควรระวัง:❌ หลีกเลี่ยงการใช้ในผู้ที่แพ้ส่วนประกอบของยา\n❌ หากมีอาการแพ้ เช่น ผื่นคัน หายใจลำบาก ควรหยุดใช้และรีบพบแพทย์\n❌ หากอาการไอไม่ดีขึ้นภายใน 7 วัน หรือมีไข้ร่วมด้วย ควรปรึกษาแพทย์\n"
                
            label_3 = tk.Label(popup2, text=info_text_2, font=("Arial", 12))
            label_3.pack()
            btn_close2 = tk.Button(popup2,text="Close",command=popup2.destroy)
            btn_close2.pack(pady=10)

            # x,y = popup.winfo_pointerxy()
            # dropdown_menu.post(x,y)
            # dropdown_menu.post(popup.winfo_pointerx(), popup.winfo_pointery())
            # messagebox.showinfo("Detected Information",info_text_2)
        for cls , conf in detection_data:
            btn = tk.Button(button_frame,text=cls,font=("Arial",12), command=lambda c=cls , f=conf: show_dropdown(c,f))
            btn.pack(side=tk.LEFT,padx=5)
        
        # for cls,conf ,_ in detection_data:
        #     info_text+=f"{cls}:{conf:.2f}\n"
        # info_label = tk.Label(popup,text=info_text,font=("Arial",14))
        # info_label.pack()
        btn_close = tk.Button(popup,text="Close",command=popup.destroy)
        btn_close.pack(pady=10)

        

# Create Tkinter window
root = tk.Tk()
# Center the window
window_width = 1024
window_height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width / 2) - (window_width / 2)
y_coordinate = (screen_height / 2) - (window_height / 2)
root.geometry(f'{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}')
root.iconbitmap("icon\\smart_pharm_zRi_icon.ico")
app = CameraApp(root, "Smart pharm")



