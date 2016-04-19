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
		self._docs = docs
		self.ct = 0
		self._iterations = iterations
		self._parsed = parsed

		self._a = 1
		self._b = self._omega
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
		xi_d = n.zeros((self._T, self._K))
		for i in range(self._T):
			for j in range(self._K):
				xi_aux = 0.
				for k in range(N_d):
					xi_aux += self._Elogbeta[j, newdoc[k]]
				xi_d[i, j] = n.exp(xi_aux)

		#init phi
		phi_d = n.zeros((N_d, self._T))
		for i in range(N_d):
			for j in range(self._T):
				phi_aux = 0.
				for k in range(self._K):
					phi_aux += xi_d[j, k] * self._Elogbeta[k, newdoc[i]]
				phi_d = n.exp(phi_aux)

		gamma_a_old = n.random.gamma(100., 1./100., (self._T))
		gamma_b_old = n.random.gamma(100., 1./100., (self._T))
		gamma_a = n.zeros(self._T)
		gamma_b = n.zeros(self._T)

		#run iterations
		for i in range(self._iterations):
			for j in range(self._T):
				gamma_a[j] = 1 + n.sum(phi_d[:, j])
				gamma_b[j] = self._alpha + n.sum(phi_d[:, j+1:])
				xi_aux = beta_expectation(self._a, self._b, self._K)
				for k in range(self._K):
					phibeta_aux = n.zeros(N_d)
					for nn in range(N_d):
						phibeta_aux[nn] = phi_d[nn, j]*self._Elogbeta[k, newdoc[nn]]
					xi_d[j, k] = n.exp(xi_aux[k]) + n.sum(phibeta_aux)

			for j in range(N_d):
				phi_aux = beta_expectation(gamma_a, gamma_b, self._T)
				for k in range(self._T):
					xibeta_aux = n.zeros(self._K)
					for nn in range(self._K):
						xibeta_aux[nn] = xi_d[k, nn]*self._Elogbeta[nn, newdoc[j]]
					phi_d[j, k] = n.exp(phi_aux[k]) + n.sum(xibeta_aux)

			meanchange = n.mean(abs(gamma_b_old - gamma_b)) + n.mean(abs(gamma_a_old - gamma_a))
			if meanchange < meanchangethresh:
				break
			gamma_a_old = gamma_a
			gamma_b_old = gamma_b

		return xi_d, phi_d, newdoc

	def updateGlobal(self, xi, phi, doc):
		#set intermediae param
		lambda_new = n.zeros(self._K, self._V)
		a_new = n.zeros(self._K)
		b_new = n.zeros(self._K)

		#compute intermediate topics
		for k in range(self._K):
			xi_aux_lambda = 0.

			for t in range(self._T):
				phi_dt = n.zeros(self._V)
				for m, word in enumerate(doc):
					phi_dt[word] += phi[m, t]
				xi_aux_lambda += phi_dt * xi[t, k]

			lambda_new[k, :] = self._eta + self._D * xi_aux_lambda
			a_new[k] = 1 + self._D * n.sum(xi[:, k])
			b_new[k] = self._omega + self._D* n.sum(xi[:, k+1:self._K])

		#set rho 
		rho = (self.ct + self._tau) **(-self._kappa)

		#update
		self._lambda = (1-rho) * self._lambda + rho * lambda_new
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


