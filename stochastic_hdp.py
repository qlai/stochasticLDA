import sys, re, time, string, random, csv, argparse
import numpy as n
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize
from utils import *
# import matplotlib.pyplot as plt

n.random.seed(10000001)
meanchangethresh = 1e-3
MAXITER = 10000


class SVIHDP():
	def __init__(self, vocab, K, D, T, alpha, eta, omega, tau, kappa, docs, iterations):
		self._vocab = vocab
		self._V = len(vocab)
		self._K = K #corpus level truncation
		self._D = D
		self._T = T # document level truncation
		self._alpha = alpha
		self._eta = eta
		self._omega = omega
		self._tau = tau
		self._kappa = kappa
		self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		# print self._expElogbeta
		self._docs = docs
		self.ct = 0
		self._iterations = iterations

		self._a = n.ones(self._K)
		self._b = n.ones(self._K) * self._omega
		# print self._lambda.shape

	def updateLocal(self, doc):
		(words, counts) = doc
		newdoc = []
		N_d = sum(counts)
		for i, item in enumerate(counts):
			for j in range(item):
				newdoc.append(words[i])
		assert len(newdoc) == N_d, "error"

		#init xi
		# xi_d = n.zeros((self._T, self._K))
		# for i in range(self._T):
		# 	for j in range(self._K):
		# 		xi_aux = 0.
		# 		# print i, j, 'hello'
		# 		for k in range(N_d):
		# 			xi_aux += self._expElogbeta[j, newdoc[k]]
		# 			# print xi_aux
		# 		xi_d[i, j] = n.exp(xi_aux)
		# # print xi_d
		# phi_d = n.zeros((N_d, self._T))
		# for i in range(N_d):
		# 	for j in range(self._T):
		# 		phi_aux = 0.
		# 		for k in range(self._K):
		# 			# print xi_d[j, k], self._Elogbeta[k, newdoc[i]]
		# 			phi_aux += xi_d[j, k] * self._Elogbeta[k, newdoc[i]]
		# 		# print phi_aux
		# 		phi_d[i, j] = n.exp(phi_aux)
		# # print phi_d

		xi_dd = n.zeros((self._T, self._K, N_d))
		phi_dd = n.zeros((self._T, self._K, N_d))

		# print xi_dd
		# print phi_dd

		#init
		for i in range(self._T):
			for k in range(self._K):
				for nn in range(N_d):
					xi_dd[i, k, nn] = self._Elogbeta[k, newdoc[nn]]
		# print xi_dd
		# print n.sum(xi_dd, axis = 2)
		xi_d = n.exp(n.sum(xi_dd, axis = 2)/100)
		# print xi_d.shape
		# print xi_d

		for i in range(self._T):
			for k in range(self._K):
				for nn in range(N_d):
					phi_dd[i, k, nn] = xi_d[i, k] * self._Elogbeta[k, newdoc[nn]]


		phi_d= n.exp(n.sum(phi_dd, axis = 1)) 
		# print phi_d.shape, phi_d

		gamma_a_old = n.random.gamma(100., 1./100., (self._T))
		gamma_b_old = n.random.gamma(100., 1./100., (self._T))
		gamma_a = n.zeros(self._T)
		gamma_b = n.zeros(self._T)



		#run iterations
		for i in range(self._iterations):

			phi_d_aux = n.zeros((self._T, N_d))
			for i in range(self._T-1):
				for nn in range(N_d):
					# print phi_d[i+1:self._T, nn]
					phi_d_aux[i, nn] = n.sum(phi_d[i+1:self._T, nn], 0)
			xi_aux = beta_expectation(self._a, self._b, self._K)

			for i in range(self._T):
				gamma_a = 1 + n.sum(phi_d, 1)
				gamma_b = self._alpha + n.sum(phi_d_aux, 1)

				for k in range(self._K):
					for nn in range(N_d):
						xi_dd[i, k, nn] = phi_d[i, nn] * self._Elogbeta[k, newdoc[nn]]
					
					xi_d[i, k] = n.exp(xi_aux[k]+n.sum(xi_dd[i][k]))

			phi_aux = beta_expectation(gamma_a, gamma_b, self._T)
			for i in range(self._T):
				for nn in range(N_d):
					for k in range(self._K):
						phi_dd[i, k, nn] = xi_d[i, k] * self._Elogbeta[k, newdoc[nn]]
					phi_d[i, nn] = n.exp(phi_aux[i] + n.sum(phi_dd[i, :, nn]))
			meanchange = n.mean(abs(gamma_b_old - gamma_b)) + n.mean(abs(gamma_a_old - gamma_a))
			if meanchange < meanchangethresh:
				break
			gamma_a_old = gamma_a
			gamma_b_old = gamma_b

		return xi_d, phi_d, newdoc

	def updateGlobal(self, xi, phi, doc):
		#set intermediae param
		lambda_new = n.zeros((self._K, self._V))
		a_new = n.zeros(self._K)
		b_new = n.zeros(self._K)
		b_aux = n.zeros((self._T, self._K))
		#compute intermediate topics
		for k in range(self._K):
			lambda_aux = n.zeros((self._T,self._V))
			for i in range(self._T):
				for nn in range(len(doc)):
					lambda_aux[i, doc[nn]] += phi[i, nn]
			lambda_new[k] = self._eta + self._D * n.dot(xi[:, k], lambda_aux)
			a_new = 1 + self._D * n.sum(xi[:, k])
			if k < self._K - 1:
				b_aux[i, k] = n.sum(xi[i, k+1:self._K])
			# print n.sum(b_aux, 0)
			b_new = self._omega + self._D * n.sum(b_aux, 0)


		#set rho 
		rho = (self.ct + self._tau) **(-self._kappa)
		# print lambda_new
		# print rho
		#update
		self._lambda = (1-rho) * self._lambda + rho * lambda_new
		# print self._lambda
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._a = (1-rho) * self._a + rho * a_new
		self._b = (1-rho) * self._b + rho * b_new


	def runHDP(self):
		for it in range(self._iterations):
			randint = random.randint(0, self._D-1)
			print "ITERATION ", it, " running doc number ", randint
			doc = parseDocument(self._docs[randint], self._vocab)
			xi, phi, newdoc = self.updateLocal(doc)
			self.updateGlobal(xi, phi, newdoc)
			self.ct += 1

		# print self._lambda
		# print n.shape(self._lambda)


	def computeProbabilities(self):

		prob_topics = n.sum(self._lambda, axis = 1)
		prob_topics = prob_topics/n.sum(prob_topics)
		# print prob_topics
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
				results[i][j] = n.exp(results[i][j])
		resultsnormalizer = n.sum(results, axis = 1)
		# print resultsnormalizer
		for j in range(len(docs)):
			results[j, :] = results[j, :] / resultsnormalizer[j] 
		finalresults = n.zeros(len(docs))
		for k in range(len(docs)):
			finalresults[k] = n.argmax(results[k])
		return finalresults, prob_topics



def test():
	alldocs = getalldocs()
	vocab = getVocab("dictionary2.csv")
	test_set = SVIHDP(vocab = vocab, K = 5, D = 970, T = 5, alpha = 1, eta = 0.2, omega = 1, tau = 1024, kappa = 0.7, docs = alldocs, iterations = 100)
	test_set.runHDP()
	test_set.computeProbabilities()

	testlambda = test_set._lambda
	for k in range(0, len(testlambda)):
		lambdak = list(testlambda[k, :])
		lambdak = lambdak / sum(lambdak)
		temp = zip(lambdak, range(0, len(lambdak)))
		temp = sorted(temp, key = lambda x: x[0], reverse=True)
		# print temp
		print 'topic %d:' % (k)
		# feel free to change the "53" here to whatever fits your screen nicely.
		for i in range(0, 10):
			print '%20s  \t---\t  %.4f' % (vocab.keys()[vocab.values().index(temp[i][1])], temp[i][0])
		# print



test()
