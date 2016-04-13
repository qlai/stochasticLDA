import sys, csv
from nltk.tokenize import wordpunct_tokenize

LISTOFDOCS = "alldocs.txt"


filenames = []

def get_filenames():
	print "getting filenames"
	with open(LISTOFDOCS, 'r') as f:
		docs = f.readlines()

		for doc in docs:
			# print str(doc).split("\n")
			filenames.append(str(doc).split("\n")[0])

	return filenames


def getfiles(filename):
	# print "getting file " + filename
	f = open(filename, 'r')
	doc = f.read()
	# print doc
	return doc


def getalldocs():
	files = get_filenames()
	docs = []
	for file in files:
		doc = getfiles(file)
		# print doc
		docs.append(doc)
	# print docs
	return docs


# getalldocs()