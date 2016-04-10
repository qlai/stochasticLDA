import sys, re, time, string, random, csv
import numpy as n
from scipy.special import gammaln, psi
from nltk.tokenize import wordpunct_tokenize
from utils import *
import matplotlib.pyplot as plt

n.random.seed(10000001)
meanchangethresh = 1e-6
MAXITER = 1000


def dirichlet_expectation(alpha):
	'''see onlineldavb.py by Blei et al'''
	if len(alpha.shape) == 1:
		return (psi(alpha) - psi(n.sum(alpha)))
	return (psi(alpha) - psi(n.sum(alpha)))


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
				dictionary[wordtk] = 0
			else:
				dictionary[wordtk] += 1

	wordslist.append(dictionary.keys())
	countslist.append(dictionary.values())
	# print wordslist
	# print countslist
	wordslistVectors = []

	for word in wordslist[0]:
		# print word
		wordvector = n.zeros(len(vocab))
		wordvector[word] = 1
		# print wordvector.argmax()
		wordslistVectors.append(wordvector)

	return (wordslistVectors, countslist[0])

def getVocab(file):
	'''getting vocab dictionary from a csv file (nostopwords)'''
	vocab = dict()
	with open(file, 'r') as infile:
		reader = csv.reader(infile)
		for index, row in enumerate(reader):
			vocab[row[0]] = index

	return vocab


class SVILDA():
	def __init__(self, vocab, K, D, alpha, eta, tau, kappa, docs):
		self._vocab = vocab
		self._V = len(vocab)
		self._K = K
		self._D = D
		self._alpha = alpha
		self._eta = eta
		self._tau = tau
		self._kappa = kappa
		self._updatect = 0
		self.lambdas = []
		self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._docs = docs
		self.ct = 0
		self.lambdas = {}
		for i in range(K):
			self.lambdas[i] = []
			self.lambdas[i].append(self._lambda[i][0])
		# print self._lambda
		# print type(self._lambda)
		print self._lambda.shape


	def updateLocal(self, doc): #word_dn is an indicator variable with dimension V
		print "updating local parameters"
		(words, counts) = doc
		# print len(words), len(counts)
		newdoc = []
		# print words, counts
		N_d = sum(counts)
		phi_d = n.zeros((self._K, N_d))
		gamma_d = n.ones(self._K)
		Elogtheta_d = dirichlet_expectation(gamma_d)
		expElogtheta_d = n.exp(Elogtheta_d)
		# expElogbeta_d = self._expElogbeta(:, words)
		for i, item in enumerate(counts):
			# print i, item
			for j in range(item):
				newdoc.append(words[i])
		assert len(newdoc) == N_d, "error"

		for i in range(MAXITER):
			# print i
			for m, word in enumerate(newdoc):
				# print word, n.argmax(word)
				for k in range(self._K):

					phi_d[k][m] = expElogtheta_d[k] * self._expElogbeta[k][n.argmax(word)]
			# print phi_d
			# print n.sum(phi_d, axis = 1).size
			gamma_new = self._alpha + n.sum(phi_d, axis = 1)
			meanchange = n.mean(abs(gamma_d - gamma_new))
			# print gamma_d, gamma_new, meanchange
			# print i, meanchange
			if (meanchange < meanchangethresh):
				break

			gamma_d = gamma_new
			Elogtheta_d = dirichlet_expectation(gamma_d)
			expElogtheta_d = n.exp(Elogtheta_d)
			# expElogbeta_d = self._expElogbeta[:, words]

		newdoc = n.asarray(newdoc)
		return phi_d, newdoc

	def updateGlobal(self, phi_d, doc):
			print 'updating global parameters'
			# print phi_d
			lambda_d = n.zeros((self._K, self._V))
			for k in range(self._K):
				phi_dk = n.zeros(self._V)
				for m in range(len(doc)):
					# print phi_d[k][m], doc[m][n.argmax(doc[m])]
					phi_dk += phi_d[k][m] * doc[m] 
				
				# print n.sum(phi_dk)

				lambda_d[k] = self._eta + 1e10 * phi_dk
				# print lambda_d[k]
			rho = (self.ct + self._tau) **(-self._kappa)
			self._lambda = (1-rho) * self._lambda + rho * lambda_d
			self._Elogbeta = dirichlet_expectation(self._lambda)
			self._expElogbeta = n.exp(self._Elogbeta)
			# print self._lambda
			# print "lambda shape ", self._lambda.shape
			# self.lambdas.append(self._lambda)
			for i in range(self._K):
				self.lambdas[i].append(self._lambda[i][0])

	
	
	def runSVI(self):
		parseddocs = {}
		for i in range(MAXITER):
			
			randint = random.randint(0, self._D)
			print "ITERATION", i, " running document number ", randint
			if str(randint) not in parseddocs.keys():
				# print self._docs[randint]
				parseddocs[str(randint)] = parseDocument(self._docs[randint],self._vocab)

			doc = parseddocs[str(randint)]
			phi_doc, newdoc = self.updateLocal(parseddocs[str(randint)])
			self.updateGlobal(phi_doc, newdoc)
			self.ct += 1
		# return self.lambdas

allmydocs = getalldocs()
vocab = getVocab("dictionary.csv")

testset = SVILDA(vocab = vocab, K = 11, D = 275, alpha = 0.1, eta = 0.1, tau = 1024, kappa = 0.7, docs = allmydocs)
testset.runSVI()
yy = testset.lambdas
# print yy
finallambda = testset._lambda
# xx = n.arange(0, MAXITER+1)
# print len(xx), len(yy[0])
# # (a, b) = yy[0].shape
# for i in range(11):
# 	plt.plot(xx, yy[i])
# 	plt.show()

for i in range(11):
	bestwords = sorted(range(len(finallambda[i])), key=lambda j:finallambda[i][j])
	print i
	for k, word in enumerate(bestwords):
		print word, vocab.keys()[vocab.values().index(word)]
		if k >= 10:
			break

		
