import sys
sys.path.insert(0,'c:/Users/mkpan/Desktop/dev_area/CSDS488/ecse488-security-system/') # Had to hard code path 
from scripts import *
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from flask import Flask, redirect, url_for, render_template, Response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Camera1")
def camera1_video(): 
    return main()

if __name__ == "__main__":
    app.run(debug=True)