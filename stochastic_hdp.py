import sys, re, time, string, random, csv, argparse
import numpy as n
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize
from utilsold import *
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

		gamma_a = n.zeros(self._T)
		gamma_b = n.zeros(self._T)

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

			'''need breaking condition'''

		return xi_d, phi_d

	def updateGlobal(self, xi, phi, doc):
		#set intermediae param
		lambda_new = n.zeros(self._K)
		a_new = n.zeros(self._K)
		b_new = n.zeros(self._K)

		for k in range(self._K):
			for m, word in enumerate(doc):

			lambda_new = self._eta + self._D *
'''finish updates for a and b and lambda'''

		rho = (self.ct + self._tau) **(-self._kappa)
		self._lambda = (1-rho) * self._lambda + rho * lambda_d
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = n.exp(self._Elogbeta)
		self._a = (1-rho) * self._a + rho * a_new
		self._b = (1-rho) * self._b + rho * b_new


	def runHDP(self):