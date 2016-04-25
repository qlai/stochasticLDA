import sys, re, time, string, random, csv, argparse
import numpy as n
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize
from utils import *
# import matplotlib.pyplot as plt

n.random.seed(10000001)
meanchangethresh = 1e-3
MAXITER = 10000



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
		self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._docs = docs
		self.ct = 0
		self._iterations = iterations
		self._parsed = parsed
		print self._lambda.shape
		self._trace_lambda = {}
		for i in range(self._K):
			self._trace_lambda[i] = [self.computeProbabilities()[i]]
		self._x = [0]


	def updateLocal(self, doc): #word_dn is an indicator variable with dimension V
		(words, counts) = doc
		newdoc = []
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
			for m, word in enumerate(newdoc):
				phi_d[:, m] = n.multiply(expElogtheta_d, self._expElogbeta[:, word]) + 1e-100
				phi_d[:, m] = phi_d[:, m]/n.sum(phi_d[:, m])

			gamma_new = self._alpha + n.sum(phi_d, axis = 1)
			meanchange = n.mean(abs(gamma_d - gamma_new))
			if (meanchange < meanchangethresh):
				break

			gamma_d = gamma_new
			Elogtheta_d = dirichlet_expectation(gamma_d)
			expElogtheta_d = n.exp(Elogtheta_d)

		newdoc = n.asarray(newdoc)
		return phi_d, newdoc, gamma_d

	def updateGlobal(self, phi_d, doc):
			# print 'updating global parameters'
		lambda_d = n.zeros((self._K, self._V))

		for k in range(self._K):
			phi_dk = n.zeros(self._V)
			for m, word in enumerate(doc):
					# print word
				phi_dk[word] += phi_d[k][m] 
			lambda_d[k] = self._eta + self._D * phi_dk
		rho = (self.ct + self._tau) **(-self._kappa)
		self._lambda = (1-rho) * self._lambda + rho * lambda_d
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)

		if self.ct % 10 == 9:
			for i in range(self._K):
				self._trace_lambda[i].append(self.computeProbabilities()[i])
			self._x.append(self.ct)

	
	
	def runSVI(self):

		for i in range(self._iterations):			
			randint = random.randint(0, self._D-1)
			print "ITERATION", i, " running document number ", randint
			if self._parsed == False:
				doc = parseDocument(self._docs[randint],self._vocab)
				phi_doc, newdoc, gamma_d = self.updateLocal(doc)
				self.updateGlobal(phi_doc, newdoc)
				self.ct += 1


	def computeProbabilities(self):

		prob_topics = n.sum(self._lambda, axis = 1)
		prob_topics = prob_topics/n.sum(prob_topics)
		return prob_topics

	def getTopics(self, docs = None):
		prob_topics = self.computeProbabilities()
		prob_words = n.sum(self._lambda, axis = 0)

		if docs == None:
			docs = self._docs
		results = n.zeros((len(docs), self._K))
		for i, doc in enumerate(docs):
			parseddoc = parseDocument(doc, self._vocab)

			for j in range(self._K):
				aux = [self._lambda[j][word]/prob_words[word] for word in parseddoc[0]]
				doc_probability = [n.log(aux[k]) * parseddoc[1][k] for k in range(len(aux))]
				results[i][j] = sum(doc_probability) + n.log(prob_topics[j])
		finalresults = n.zeros(len(docs))
		for k in range(len(docs)):
			finalresults[k] = n.argmax(results[k])
		return finalresults, prob_topics

	def calcPerplexity(self, docs = None):
		perplexity = 0.
		doclen = 0.
		if docs == None:
			docs =  self._docs
		for doc in docs:
			parseddoc = parseDocument(doc, self._vocab)
			_, newdoc, gamma_d = self.updateLocal(parseddoc)
			approx_mixture = n.dot(gamma_d, self._lambda)
			# print n.shape(approx_mixture)
			approx_mixture = approx_mixture / n.sum(approx_mixture)
			log_doc_prob = 0.
			for word in newdoc:
				log_doc_prob += n.log(approx_mixture[word])
			perplexity += log_doc_prob
			doclen += len(newdoc)
			# print perplexity, doclen
		perplexity = n.exp( - perplexity / doclen)
		print perplexity
		return perplexity

	def plotTopics(self, perp):
		plottrace(self._x, self._trace_lambda, self._K, self._iterations, perp)
			

			



def test(k, iterations):

	allmydocs = getalldocs("alldocs2.txt")
	vocab = getVocab("dictionary2.csv")
	testset = SVILDA(vocab = vocab, K = k, D = 847, alpha = 0.2, eta = 0.2, tau = 1024, kappa = 0.7, docs = allmydocs, iterations= iterations)
	testset.runSVI()

	finallambda = testset._lambda
	
	heldoutdocs = getalldocs("testdocs.txt")

	perplexity = testset.calcPerplexity(docs = heldoutdocs)

	with open("temp/%i_%i_%f_results.csv" %(k, iterations, perplexity), "w+") as f:
		writer = csv.writer(f)
		for i in range(k):
			bestwords = sorted(range(len(finallambda[i])), key=lambda j:finallambda[i, j])
			# print bestwords
			bestwords.reverse()
			writer.writerow([i])
			for j, word in enumerate(bestwords):
				writer.writerow([word, vocab.keys()[vocab.values().index(word)]])
				if j >= 15:
					break
	topics, topic_probs = testset.getTopics()
	testset.plotTopics(perplexity)

	for kk in range(0, len(finallambda)):
		lambdak = list(finallambda[kk, :])
		lambdak = lambdak / sum(lambdak)
		temp = zip(lambdak, range(0, len(lambdak)))
		temp = sorted(temp, key = lambda x: x[0], reverse=True)
		# print temp
		print 'topic %d:' % (kk)
		# feel free to change the "53" here to whatever fits your screen nicely.
		for i in range(0, 10):
			print '%20s  \t---\t  %.4f' % (vocab.keys()[vocab.values().index(temp[i][1])], temp[i][0])
		print


	with open("temp/%i_%i_%f_raw.txt" %(k, iterations, perplexity), "w+") as f:
		# f.write(finallambda)
		for result in topics:
			f.write(str(result) + " \n")
		f.write(str(topic_probs) + " \n")

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
