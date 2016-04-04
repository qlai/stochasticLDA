import sys, re, time, string, random, csv
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(10000001)
meanchangethresh = 1e-3
MAXITER = 1000


def dirichlet_expectation(alpha):
	'''see onlineldavb.py by Blei et al'''
	if len(alpha.shape) == 1:
		return (psi(alpha) - psi(n.sum(alphs)))
	return (psi(alpha) - psi(n.sum(alphs))[:, n.newaxis])


def parseDocument(doc, vocab):
	wordslist = list()
	countslist = list()
	doc = re.sub(r'-', ' ', doc)
	doc = re.sub(r'[^a-z]', '', doc)
	doc = re.sub(r' +', ' ', doc)
	words = string.split(doc)

	dictionary = dict()
	for word in words:
		if word in vocab:
			wordtk = vocab[word]
			if wordtk not in dictionary:
				dictionary[wordtk] = 0
			else:
				dictionary[wordtk] += 1

	wordslist.append(dictionary.keys())
	countslist.append(dictionary.values())

	wordslistVectors = list()

	for word in wordslist:
		wordvector = np.zeros(len(vocab))
		wordvector[word] = 1
		wordslistVectors.append(wordvector)

	return (wordslistVectors, countslist)

def getVocab(file):
	'''getting vocab dictionary from a csv file (nostopwords)'''
	vocab = dict()
	with open(file, 'r') as infile:
		reader = csv.reader(infile)
		for index, row in enumerate(reader):
			vocab[row] = index

	return vocab


class SVILDA():
	def __init__(self, vocab, K, D, alpha, eta, rho, docs):
		self._vocab = vocab
		self._V = len(vocab)
		self._K = K
		self._D = D
		self._alpha = alpha
		self._eta = eta
		self._rho = rho
		self._updatect = 0
		self.lambdas = []
		self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._docs = docs
		self.lambdas.append(self._lambda)


	def updateLocal(self, doc): #word_dn is an indicator variable with dimension V
		(words, counts) = doc
		newdoc = []
		N_d = sum(counts)
		phi_d = np.zeros((self._K, N_d))
		gamma_d = n.ones(self._K)
		Elogtheta_d = dirichlet_expectation(gamma_d)
		expElogtheta_d = n.exp(Elogtheta)
		# expElogbeta_d = self._expElogbeta(:, words)
		for i, item in enumerate(counts):
			for j in range(item):
				newdoc.append(item)
		assert len(newdoc) == N_d, "error"

		for i in range(MAXITER):
			for n, word in enumerate(newdoc):
				for k in range(self._K):

					phi_d[k][n] = expElogtheta_d[k] * self._expElogbeta_d[k][word]

			gamma_new = self._alpha + n.sum(phi_d, axis = 1)
			meanchange = n.mean(abs(gamma_d - gammanew))
			if (meanchange < meanchangethresh):
				break

			gamma_d = gammanew
			Elogtheta_d = dirichlet_expectation(gamma_d)
			expElogtheta_d = n.exp(Elogtheta)
			# expElogbeta_d = self._expElogbeta[:, words]

		newdoc = np.asarray(newdoc)
		return phi_d, newdoc

	def updateGlobal(self, phi_d, doc):(
			lambda_d = n.zeros(self._K, self._V)
	
			for k in range(self._K):
				phid_dk = np.zeros(self._V)
				for n in range(len(doc)):
					phi_dk += phi_d[k][n] * doc[n] 
				
				lambda_d[k] = self._eta + self._D * phi_dk
	
			self._lambda = (1-self._rho) * self._lambda + self._rho * self.lambda_d
			self._Elogbeta = dirichlet_expectation(self._lambda)
			self._expElogbeta = n.exp(self._Elogbeta)
	
			self.lambdas.append(self._lambda)

	
	
		def runSVI(self):
			parseddocs = {}
			while i in range(MAXITER):
				randint = random.randint(0, self._D)
				if not parseddocs[self._docs[randint]]:
					parseDocument(docs[randint], self._vocab)
				doc = parseddocs[self._docs[randint]]
				phi_doc, newdoc = self.updateLocal(parseddocs[self._docs[randint]])
				self.updateGlobal(phi_doc, newdoc)








		
