import sys, csv
from nltk.tokenize import wordpunct_tokenize
from scipy.special import psi
import numpy as n

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



def dirichlet_expectation(alpha):
	'''see onlineldavb.py by Blei et al'''
	if (len(alpha.shape) == 1):
		return (psi(alpha) - psi(n.sum(alpha)))
	return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def beta_expectation(V, a, b, k):
	Elog_a = psi(a) - psi(a + b)
	Elog_b = psi(b) - psi(b + a)
	Elog_sig = n.zeros(k)
	for l in range(1, k+1):
		Elog_sig[l] = Elog_a[l] + n.sum(Elog_b[0:l-1])
	return Elog_sig


def parseDocument(doc, vocab):
	wordslist = list()
	countslist = list()
	doc = doc.lower()
	tokens = wordpunct_tokenize(doc)

	dictionary = dict()
	for word in tokens:
		if word in vocab:
			wordtk = vocab[word]
			if wordtk not in dictionary:
				dictionary[wordtk] = 1
			else:
				dictionary[wordtk] += 1

	wordslist.append(dictionary.keys())
	countslist.append(dictionary.values())
	return (wordslist[0], countslist[0])

def getVocab(file):
	'''getting vocab dictionary from a csv file (nostopwords)'''
	vocab = dict()
	with open(file, 'r') as infile:
		reader = csv.reader(infile)
		for index, row in enumerate(reader):
			vocab[row[0]] = index

	return vocab




# getalldocs()