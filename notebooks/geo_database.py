from pymongo import MongoClient, GEO2D
import pymongo
import pymongo.errors
from geojson import Feature
import pprint
import os
from bson.son import SON


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


def add(record):
	parking_collection = get_collection()
	try:
		result = parking_collection.insert(record)
	except pymongo.errors.DuplicateKeyError:
		pass


def addMany():
	parking_collection = get_collection()
	try:
		result = parking_collection.insert_many([
			{type:"parking",'loc':[2,5]},
			{type:"parking",'loc':[30,5]},
			{type:"parking",'loc':[1,2]},
			{type:"parking",'loc':[4,4]}
		])
		print result.inserted_ids
	except pymongo.errors.DuplicateKeyError:
		pass


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


def dedupe(matches):
    """ Some of our records are duplicated, this fixes that."""
    seen_coords = set()

    for match in matches:
        coords = tuple(match['loc']['coordinates'])

        if coords not in seen_coords:
            seen_coords.add(coords)
            yield match

#add()
findWithInRec([2,2],[5,6])
findWithInCircle([0,0],6)
#find(2,2)



def find(lng, lat, within=250, type="parking", limit=100, collection=None):

	parking_collection = get_collection()	

	matches = parking_collection.find({
		"type": type,
		"loc": {
		"$near": {
			"$geometry":
				{
					"type": "Point",
					"coordinates": [lng, lat],
				},
				"$maxDistance": within,  # measured in meters
			}
		},
	}, {'_id': False}).limit(limit)

	return list(dedupe(matches))


def load_data(): 

	collection = get_collection()

	with open('data/bikes.json', 'r') as f:
		bikes = json.load(f)

	columns = [c['fieldName'] for c in bikes['meta']['view']['columns']]

	for bike in bikes['data']:
		bike_data = dict(zip(columns, bike))

	if bike_data['status'] != 'COMPLETE':
		continue

	bike_record = dict(
		type='bike',
		loc=dict(
			type='Point',
			coordinates=[
				float(bike_data['coordinates'][2]),  # long
				float(bike_data['coordinates'][1]),  # lat
			]
		)
	)

	try:
		collection.insert(bike_record)
		except pymongo.errors.DuplicateKeyError:
	pass