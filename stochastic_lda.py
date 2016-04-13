import sys, re, time, string, random, csv, argparse
import numpy as n
from scipy.special import gammaln, psi
from nltk.tokenize import wordpunct_tokenize
from utilsold import *
# import matplotlib.pyplot as plt

n.random.seed(10000001)
meanchangethresh = 1e-10
MAXITER = 10000


def dirichlet_expectation(alpha):
	'''see onlineldavb.py by Blei et al'''
	# print alpha.shape
	if (len(alpha.shape) == 1):
		return (psi(alpha) - psi(n.sum(alpha)))
	# print (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])
	return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


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


class SVILDA():
	def __init__(self, vocab, K, D, alpha, eta, tau, kappa, docs, iterations, parsed = False):
		self._vocab = vocab
		self._V = len(vocab)
		self._K = K
		self._D = D
		self._alpha = alpha
		self._eta = eta
		self._tau = tau
		self._kappa = kappa
		# self._updatect = 0
		self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._docs = docs
		self.ct = 0
		self._iterations = iterations
		self._parsed = parsed
		print self._lambda.shape


	def updateLocal(self, doc): #word_dn is an indicator variable with dimension V
		# print "updating local parameters"
		(words, counts) = doc
		# print len(words), len(counts)
		newdoc = []
		# print words, counts
		N_d = sum(counts)
		phi_d = n.zeros((self._K, N_d))
		gamma_d = n.random.gamma(100., 1./100., (self._K))
		Elogtheta_d = dirichlet_expectation(gamma_d)
		expElogtheta_d = n.exp(Elogtheta_d)
		for i, item in enumerate(counts):
			for j in range(item):
				newdoc.append(words[i])
		assert len(newdoc) == N_d, "error"

		for i in range(self._iterations):
			# print self._expElogbeta[:, 0], expElogtheta_d
			for m, word in enumerate(newdoc):
				phi_d[:, m] = n.dot(expElogtheta_d, self._expElogbeta[:, word])
			# print n.sum(phi_d, axis = 1)
			# print phi_d
			gamma_new = self._alpha + n.sum(phi_d, axis = 1)
			# print self._alpha, gamma_new, gamma_d
			meanchange = n.mean(abs(gamma_d - gamma_new))
			if (meanchange < meanchangethresh):
				break

			gamma_d = gamma_new
			Elogtheta_d = dirichlet_expectation(gamma_d)
			expElogtheta_d = n.exp(Elogtheta_d)

		# print gamma_new
		# print N_d, phi_d
		newdoc = n.asarray(newdoc)
		return phi_d, newdoc

	def updateGlobal(self, phi_d, doc):
			# print 'updating global parameters'
			lambda_d = n.zeros((self._K, self._V))

			for k in range(self._K):
				phi_dk = n.zeros(self._V)

				for m, word in enumerate(doc):
					# print word
					phi_dk[word] += phi_d[k][m] 

				lambda_d[k] = self._eta + self._D * phi_dk
			# print lambda_d
			rho = (self.ct + self._tau) **(-self._kappa)
			# print 'rho', rho
			self._lambda = (1-rho) * self._lambda + rho * lambda_d
			# print self._lambda
			self._Elogbeta = dirichlet_expectation(self._lambda)
			self._expElogbeta = n.exp(self._Elogbeta)

	
	
	def runSVI(self):

		for i in range(self._iterations):			
			randint = random.randint(0, self._D-1)
			print "ITERATION", i, " running document number ", randint
			if self._parsed == False:
				doc = parseDocument(self._docs[randint],self._vocab)
				phi_doc, newdoc = self.updateLocal(doc)
				self.updateGlobal(phi_doc, newdoc)
				self.ct += 1
			# if self._parsed == True:
			# 	doc = self._docs[0][randint]



	def getTopics(self, docs = None):
		prob_words = n.sum(self._lambda, axis = 0)
		prob_topics = n.sum(self._lambda, axis = 1)
		prob_topics = prob_topics/n.sum(prob_topics)
		print prob_topics

		if docs == None:
			docs = self._docs
		results = n.zeros((len(docs), self._K))
		for i, doc in enumerate(docs):
			parseddoc = parseDocument(doc, self._vocab)

			for j in range(self._K):
				aux = [self._lambda[j][word]/prob_words[word] for word in parseddoc[0]]
				# print aux
				doc_probability = [n.log(aux[k]) * parseddoc[1][k] for k in range(len(aux))]
				# print doc_probability
				results[i][j] = sum(doc_probability) + n.log(prob_topics[j])
		finalresults = n.zeros(len(docs))
		for k in range(len(docs)):
			finalresults[k] = n.argmax(results[k])
		# print finalresults
		return finalresults

			



def test(k, iterations):

	allmydocs = getalldocs()
	vocab = getVocab("dictionary2.csv")
	testset = SVILDA(vocab = vocab, K = k, D = 941, alpha = 0.2, eta = 0.2, tau = 1024, kappa = 0.7, docs = allmydocs, iterations= iterations)
	testset.runSVI()

	finallambda = testset._lambda

	with open("results.csv", "w+") as f:
		writer = csv.writer(f)
		for i in range(k):
			bestwords = sorted(range(len(finallambda[i])), key=lambda j:finallambda[i, j])
			writer.writerow([i])
			for k, word in enumerate(bestwords):
				writer.writerow([word, vocab.keys()[vocab.values().index(word)]])
				if k >= 15:
					break
	topics = testset.getTopics()


	# with open("raw.txt", "w+") as f:
	# 	# f.write(finallambda)
	# 	for result in topics:
	# 		f.write(str(result) + " \n")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-K','--topics', help='number of topics, defaults to 10',required=True)
	parser.add_argument('-m','--mode', help='mode, test | normal',required=True)
	parser.add_argument('-v','--vocab', help='Vocab file name, .csv', default = "dictionary.csv", required=False)
	parser.add_argument('-d','--docs', help='file with list of docs, .txt', default = "alldocs.txt", required=False)
	parser.add_argument('-a','--alpha', help='alpha parameter, defaults to 0.2',default = 0.2, required=False)
	parser.add_argument('-e','--eta', help='eta parameter, defaults to 0.2',default= 0.2, required=False)
	parser.add_argument('-t','--tau', help='tau parameter, defaults to 0.7',default= 0.7, required=False)
	parser.add_argument('-k','--kappa', help='kappa parameter, defaults to 1024',default = 1024, required=False)
	parser.add_argument('-n','--iterations', help='number of iterations, defaults to 10000',default = 10000, required=False)
	
	args = parser.parse_args()

	#print args.output

	mode = str(args.mode)
	vocab = str(args.vocab)
	K = int(args.topics)
	alpha = float(args.alpha)
	eta = float(args.eta)
	tau = float(args.tau)
	kappa = float(args.kappa)
	iterations = int(args.iterations)
	docs = str(args.docs)
	vocab = str(args.vocab)

	if mode == "test":
		test(K, iterations)
	if mode == "normal":
		assert vocab is not None, "no vocab"
		assert docs is not None, "no docs"
		D = len(docs)
		docs = getalldocs(docs)
		vocab = getVocab(vocab)
		lda = SVILDA(vocab = vocab, K = K, D = D, alpha = alpha, eta = eta, tau = tau, kappa = kappa, docs = docs, iterations = iterations)
		lda.runSVI()

		return lda 


if __name__ == '__main__':
	main()
