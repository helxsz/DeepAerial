from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient, GEO2D
import pymongo
import pymongo.errors
from bson import json_util
import os
import pprint

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


def get_collection(**config):
	"""Use this to talk to Mongo.
	With no args this will default to your local Mongo instance (using the
	'parkit' database). Supply the MONGOHQ_URL in your environment to change
	this.
	"""
	mongo_url = os.getenv(
		'MONGOHQ_URL',
		'mongodb://localhost:27017/parkit'
	)
	client = MongoClient()
	db = client.parking2
	#db.authenticate('root','root')
	parking_collection = db.parking
	parking_collection.create_index([("loc",GEO2D)])
	return parking_collection

def findWithInRec( box_left, box_right):

	parking_collection = get_collection()

	query = {"loc":{"$within":{"$box":[box_left,box_right]}}}
	docs = []
	for doc in parking_collection.find(query).sort('_id'):
		pprint.pprint(doc)
		docs.append(doc)
	return docs



def findWithInCircle(center, radius):

	parking_collection = get_collection()

	query = {"loc":{"$within":{"$center":[center,radius]}}}
	docs = []
	for doc in parking_collection.find(query).sort('_id'):
		pprint.pprint(doc)
		docs.append(doc)
	return docs



@app.route('/near')
def near():
	"""Respond with X bike parkings closest to given location"""
	args = request.args

	if 'limit' in args: limit = int(args.get('limit'))
	else: limit = 1

	if 'lat' in args and 'lng' in args:
		lat = float(args.get('lat'))
		lng = float(args.get('lng'))

	else:
		return jsonify(success=False, reason='wrong_arguments')

	docs = findWithInCircle([lat,lng],6)

	return json_util.dumps({
		'success': True, 'docs': docs
	})



#http://127.0.0.1:5000/parse?index=1
@app.route('/parse')
def parseImage():
	base_dir = './'
	args = request.args

	if 'index' in args: index = int(args.get('index'))
	else: index = 1
	try:
		image = cv2.imread(base_dir +'/' + 'segnet_vaihingen_128x128_64_area{}.tif'.format(index))
	except:
		return json_util.dumps({
				'success': False
		}) 

	if image is None:
		return json_util.dumps({
				'success': False
		}) 

	boundaries= ([30, 100, 100 ],[32, 255, 255 ]) 
	image, output = parseImage(image,boundaries,False)

	cv2.imwrite( './temp.png', np.hstack([cv2.resize(image,(400,600)), cv2.resize(output,(400,600))])  )
	return send_file( './temp.png' ,mimetype='image/png')
	'''''
	return json_util.dumps({
		'success': True, 'docs': docs
	})
	'''''


def parseImage(image, boundary , show):

	lower, upper= boundary[0], boundary[1]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")


	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)	
	mask = cv2.inRange(hsv, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_NONE)
	positions = []
	for i in range(len(contours)):
		if len(contours[i]) > 20:
			x, y, w, h = cv2.boundingRect(contours[i])
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			print 'found the car at position:', (x, y),(x+w, y+h)            
			positions.append([x, y, x+w, y+h])
	#cv2.imshow("mask",mask)	
	if(show):
		
		cv2.imshow("window", np.hstack([cv2.resize(image,(400,600)), cv2.resize(output,(400,600))]))
		cv2.waitKey(0)
	#return positions
	return image, output




if __name__ == "__main__":
    app.run()

# https://github.com/carolinevdh/pedalpark/blob/2e898996f807ac85b33f95d848d9e3bf2937dac5/pedalpark/location.py