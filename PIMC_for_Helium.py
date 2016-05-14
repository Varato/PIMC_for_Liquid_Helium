# by xinchen
# 20160404
# last update 20160514
'''
Compute Superfluid Density Using PIMC.
Properties For Helium Liquid:
\epsilon = 10.22 K
\sigma = 2.556 \AA
\hbar^2/m = 12.120904 K \AA^2
density n = 1.88 e-2 \AA^-3 = 0.314 (with \sigma = 1)
'''


import numpy as np
from numpy import linalg as la
import random
from copy import deepcopy
import time
import sys
import logging  
import logging.handlers  
import cPickle as pickle
from pprint import pprint
import matplotlib.pyplot as plt

d = 3                              # Spacial Dimension
epsilon = 5.5086                   # In natural Units' system
m_staging = 10                     # Staging length
N = 16                             # Number of particles
M = 20                             # Number os time slices
MCSteps = 3000                     # Number of Mento Carlo steps
ThermalSteps = 100                 # Number of thermalizing steps
observeSkip = 10                   # Steps interval for observing winding number
L = [pow(N/0.314, 0.333)]*3        # the scale of the simulation box
# Suppose there is periodic boundary conditions along x, y, z directions. 
# ================================================================================================
# Construct a Logger 
PICKLE_FILE = 'cfg.pkl'
LOG_FILE = 'mylog0.log'  
  
fh = logging.FileHandler(LOG_FILE)
ch = logging.StreamHandler()
fmt = '%(asctime)s - %(filename)s: %(name)s - %(message)s' 
  
formatter = logging.Formatter(fmt) 
fh.setFormatter(formatter)
ch.setFormatter(formatter) 
  
logger = logging.getLogger('MCProcess')  
logger.addHandler(fh)
logger.addHandler(ch)          
logger.setLevel(logging.INFO)  

class bead:
	def __init__(self, r0, t, i):
		self.r = r0
		# These 2 pointer will change after a permutation operator
		self.next = None
		self.prev = None
		# These 2 pointer will not change
		self.ownnext = None
		self.ownprev = None
		self.t = t
		self.belongsto = i
		self.nextShift = np.array([0,0,0])
	def Next(self, s):
		tmp = self
		for i in range(s):
			tmp = tmp.next
		return tmp

# ================================================================================================
def whichPeriod(r):
	'''
	Tells which period is r at.
	'''
	n = np.zeros(d, dtype = int)
	for dd in range(d):
		while 1:
			if n[dd]*L[dd] < r[dd] <= (n[dd]+1)*L[dd]:
				break
			elif r[dd] >= 0: n[dd] += 1
			else: n[dd] -= 1
	return n

class path:

	def __init__(self, r0, sigma2, numTimeSlices, i):
		'''
		Generate a world line rooted at r0
		numTimeSlices: the number of beads of a particle
		'''
		self.numTimeSlices = numTimeSlices
		self.r0 = r0
		self.origin = None

		# initialize paths with random r0s
		# data structure: linked list
		for j in range(numTimeSlices):

			if j == 0:
				tmp = bead(r0, j, i)
				tmp0 = tmp
				self.origin = tmp
			else:
				g = np.random.normal(loc = 0.0, scale = 1.0, size = None)
				tmp = bead(tmp0.r+np.sqrt(sigma2)*g, j, i)
				# modify for PBC
				for dd in range(d):
					if tmp.r[dd] > L[dd]: tmp.r[dd] = L[dd] - 0.05
					elif tmp.r[dd] < 0: tmp.r[dd] = 0.05

				tmp.prev = tmp0
				tmp.ownprev = tmp0
				tmp0.next = tmp
				tmp0.ownnext = tmp
				tmp0 = tmp
							
				if j == numTimeSlices-1:
					tmp.next = self.origin
					tmp.ownnext = self.origin
					self.origin.prev = tmp
					self.origin.ownprev = tmp

	def getBead(self, k):
		'''
		Gets the #k bead of a path, with k starting from zero
		'''
		tmp = self.origin
		for j in range(k):
			tmp = tmp.next
		return tmp

	def getOwnBead(self, k):
		rank = k % self.numTimeSlices
		tmp = self.origin
		for j in range(rank):
			tmp = tmp.ownnext
		return tmp

# ================================================================================================

class configuration:
	
	def __init__(self, numParticles, numTimeSlices, Temperature, lam):
		'''
		numParticles: the number of real particles
		numTimeSlices: the M dividing beta (number of segments)
		lam = \hbar^2/2m = 0.5 with m=hbar=1
		'''
		self.particles = []
		self.numParticles = numParticles
		self.numTimeSlices = numTimeSlices
		self.Temperature = Temperature
		self.tau = 1.0/(Temperature*numTimeSlices)
		self.lam = lam
		sigma2 = 2*self.lam*self.tau
		for i in range(numParticles):
			r0 = np.random.uniform(0.5, L[0]-0.5, d)
			self.particles.append( path(r0, sigma2, numTimeSlices, i) )
		self.particles = tuple(self.particles)
		
	def __LennardJones(self,bead1,bead2):
		'''
		Calculates Lennard Jones potential of 2 beads
		If the 2 beads are too close, repel them
		'''
		while 1:
			deltr = bead1.r-bead2.r
			for dd in range(d):
				if np.abs(deltr[dd]) > 0.5*L[dd]:
					deltr[dd] = L[dd] - deltr[dd]
			r = la.norm(deltr)
			if r >= 0.8:
				v = 4.0*epsilon*(pow(r,-12) - pow(r,-6))
				return v
			else:
				bead2.r = bead1.r + np.random.uniform(-1, 1, d)
				# Modify for PBC
				for dd in range(d):
					if bead2.r[dd] > L[dd]: bead2.r[dd] = L[dd] - 0.05
					elif bead2.r[dd] < 0: bead2.r[dd] = 0.05
		return v


	def __PotentialAction(self, k):
		'''
		Obtains the whole interacting potential action.
		k is the time slice being chosen.
		'''
		tot = 0.0
		ii=0
		for particle in self.particles:
			ii += 1
			bead01 = particle.getOwnBead(k)
			# bead02 = bead01.next
			for others in self.particles[ii:]:
				bead11 = others.getOwnBead(k)
				# bead12 = bead11.next
				# r12 = bead12.r
				tot += self.__LennardJones(bead01, bead11) #+self.__LennardJones(bead02, bead12)
		return self.tau*tot


	def ComMove(self, start, restore = False, xshift = None):
		'''
		Attempts a center of mass move.
		i is the particle index being chosen.
		'''
		# the translation vector:
		if restore:
			qshift = xshift			
			# unfold the period
			tmp = start
			p = np.zeros(d,dtype=int)
			while 1:
				tmp.r += p*L
				p += tmp.nextShift
				tmp = tmp.next
				if tmp is start: break
			# Do COM move
			tmp = start
			while 1:
				tmp.r += qshift
				tmp = tmp.next
				if tmp is start: break
			# calculate the period shifts
			tmp = start
			while 1:
				tmp.nextShift = whichPeriod(tmp.next.r) - whichPeriod(tmp.r)
				tmp = tmp.next
				if tmp is start: break
			# refold the periods:
			tmp = start
			while 1:
				tmp.r = tmp.r - whichPeriod(tmp.r)*L
				tmp=tmp.next
				if tmp is start: break
			del tmp

		else:
			delta = np.sqrt(self.lam * self.tau)
			qshift = delta*np.random.uniform(-1, 1, d)
			#record old action
			old_action = 0.0
			for j in range(self.numTimeSlices):
				old_action += self.__PotentialAction(j)

			# calculate the period for each bead and unfold the period
			tmp = start
			p = np.zeros(d,dtype=int)
			while 1:
				tmp.r += p*L
				p += tmp.nextShift
				tmp = tmp.next
				if tmp is start: break
			# Do COM move
			tmp = start
			while 1:
				tmp.r += qshift
				tmp = tmp.next
				if tmp is start: break
			# calculate the period shifts
			tmp = start
			while 1:
				tmp.nextShift = whichPeriod(tmp.next.r) - whichPeriod(tmp.r)
				tmp = tmp.next
				if tmp is start: break
			# refold the periods:
			tmp = start
			while 1:
				tmp.r = tmp.r - whichPeriod(tmp.r)*L
				tmp=tmp.next
				if tmp is start: break
			del tmp

			#calculate new action
			new_action = 0.0
			for j in range(self.numTimeSlices):
				new_action += self.__PotentialAction(j)

			if np.random.random() < np.exp(-(new_action - old_action)):
				# Accepted!
				return True
			else:
			 	# Rejected, restore the changes
				self.ComMove(start, restore=True, xshift = -qshift)
				return False


	def __RecursionBisection(self, bead1, bead2):
		'''
		Does bisection sample between bead1 and bead2 RECURSIVLY.
		It's not necessary that bead1 and bead2 belong to the same particle.
		Only when s = 2, 4, 8, 16.... , all the beads between bead1 and bead2 will be moved.
		Mobe without PBC modifying
		'''
		s = np.abs(bead2.t - bead1.t)
		interval = int(s/2.0)
		if s <= 1:
			return
		else:
			# Do mid move
			r_bar = 0.5*(bead1.r+bead2.r)
			s = np.abs(bead2.t - bead1.t)
			sigma2 = 2.0*self.lam*self.tau*s
			g = np.random.normal(loc = 0.0, scale = 1.0, size = d)
			bead1.Next(interval).r = r_bar + np.sqrt(sigma2)*g

			for seg in range(2):
				# Recursion here
				self.__RecursionBisection( bead1.Next(seg*interval), bead1.Next((seg+1)*interval) )


	def BisectionMove(self, bead1, bead2, onlyMove = False):
		'''
		Do Bisection Move 
		The PBC is taken into account.
		'''
		if not onlyMove:
			# record old information
			s = np.abs(bead2.t - bead1.t)
			old_action = 0.0
			for j in range(s-1):
				old_action += self.__PotentialAction(bead1.t+j+1)
			old_position = []
			old_shifts = []
			tmp = bead1
			while 1:
				old_shifts.append(deepcopy(tmp.nextShift))
				tmp = tmp.next
				if tmp is bead2: break
				else: old_position.append(deepcopy(tmp.r))

		# calculate how many time the chain between bead1 and bead2 crossed the boundary
		tmp = bead1
		deltshift = np.zeros(3, dtype = int)
		while tmp is not bead2:
			deltshift += tmp.nextShift
			tmp = tmp.next
		# instantialize a temperary bead for bisection at r0
		# r0 is the coordinate of the unfolded bead2 
		r0 = bead2.r + deltshift*L
		tmpbead = bead(r0, bead2.t, -1)
		self.__RecursionBisection(bead1, tmpbead)
		# Recalculate the period shifts
		tmp = bead1
		while tmp is not bead2:
			if tmp.next is bead2:
				rr = r0
			else:
				rr = tmp.next.r
			tmp.nextShift = whichPeriod(rr) - whichPeriod(tmp.r)
			tmp = tmp.next
		# Refold periods
		tmp = bead1.next
		while tmp is not bead2:	
			tmp.r = tmp.r - whichPeriod(tmp.r) * L
			tmp = tmp.next

		if not onlyMove:
			# Accept/Reject judge
			new_action = 0.0
			for j in range(bead2.t-bead1.t-1):
				new_action += self.__PotentialAction(bead1.t+j+1)

			if np.random.random() < np.exp(-(new_action - old_action)):
				# Accepted!
				return True
			else: 
				# Rejected and restore
				tmp = bead1
				k = 0
				while 1:
					tmp.nextShift = old_shifts[k]
					tmp = tmp.next
					if tmp is bead2: break
					else: tmp.r = old_position[k]
					k+=1
				return False


# ------------------------Permutation Sample-------------------------
	def __NearerOppsite(self, i1, i2, k, s, dd):
		'''
		Tells whether the oppsite way is nearer or not under PBC
		'''
		z1k = self.particles[i1].getOwnBead(k).r[dd]
		z2ks = self.particles[i2].getOwnBead(k+s).r[dd]
		deltz = z1k - z2ks
		if np.abs(deltz) > 0.5*L[dd]:
			return True
		else:
			return False

	def __TransitionProb(self, i1, i2, k, s):
		'''
		Calculates permutation transition probability between particle i1 and i2 in time window from k to k+s.
		i1, i2 are the particle index being chosen.
		PBC should be considered.
		'''
		r1k = self.particles[i1].getOwnBead(k).r
		r2ks = self.particles[i2].getOwnBead(k+s).r
		deltr = r1k - r2ks
		# If the beads' z distance is farther than L/2, do the PBC modification:
		for dd in range(d):
			if self.__NearerOppsite(i1, i2, k, s, dd):
				if r1k[dd] <= r2ks[dd]:
					deltr[dd] = r1k[dd] + L[dd] - r2ks[dd]
				else:
					deltr[dd] = r1k[dd] - L[dd] -r2ks[dd]
		t = np.exp( -np.dot(deltr, deltr)/(4.0*s*self.lam*self.tau) )
		return t

	def __get_h(self, i0, k, s):
		'''
		Gets the normaolization parameter for particle i0
		'''
		h = 0.0
		for i in range (self.numParticles):
			t = self.__TransitionProb(i0, i, k, s)
			h += t
		return h

	def __sampleNextPtcl(self, i0, k, s):
		'''
		Samples i0's next particle for cyclic permutation
		'''
		prob = []
		hi0 = 0.0
		for i in range(self.numParticles):
			t = self.__TransitionProb(i0, i, k, s)
			prob.append(t)
			hi0 += t
		prob = np.array(prob)/hi0
 
		# does Roulette to sample the next particle
		g = random.random()
		for ii in range(len(prob)):
			if ii == 0:
				left = 0.0
				right = prob[0]
			else:
				left = sum(prob[:ii])
				right = sum(prob[:ii+1])

			if left < g <= right:
				result = ii
			
		return result

	def __connect(self, bead1, bead2, period_shift):
		'''
		connect bead1 to bead2 avoiding bad loops.
		This method can recursively restore bad loops
		'''
		bead1.nextShift = period_shift
		bead1.next = bead2
		bead2.prev = bead1

	def perm_bisecMove(self, k, s, max_sample_particles):
		'''
		Does permutaion-bisection move following instructions below:
		1. sample a particle as head 
		2. sample the other particles in sequence 
		3. Accept/Reject the cyclic permutation of these particles 
		4. If accepted, Do bisection move to regrow beads.
		'''
		logger.info('Doing Permutation...')
		# randomly select a initial ring for the cyclic permutation
		i1 = random.randint(0, self.numParticles-1)
		# sample other particles
		indexes = [i1]
		tmp1 = i1
		max_sample_steps = 100
		for ii in range(max_sample_steps):
			# if len(indexes) >= max_sample_particles:
			# 	break
			# else:
			tmp2 = self.__sampleNextPtcl(tmp1, k, s)
			if tmp2 not in indexes:
				indexes.append(tmp2)
				tmp1 = tmp2

		if len(indexes) == 1:
			logger.info('There is too low probabily to exchange.')
			return False

		# Accept or reject?
		h = []
		t_not_perm = []
		t_perm = []
		perm_indexes = indexes[1:]+[indexes[0]]
		for i in indexes:
			h.append( self.__get_h(i, k, s) )
			t_not_perm.append( self.__TransitionProb(i, i, k, s) )
		for ii in range(len(indexes)):
			t_perm.append( self.__TransitionProb(indexes[ii], perm_indexes[ii], k, s) )

		numerator = 0.0
		denominator = 0.0
		for ii in range(len(indexes)):
			numerator += h[ii]/t_not_perm[ii]
			denominator += h[ii]/t_perm[ii]
		p = numerator/denominator
		# print 'Acceptance Probability', p

		if random.random() < p:
			# Accepted! Do permutaion & Bisection move
			logger.info('Permutation Accepted!')
			logger.info('Particles being chosen: '+str(indexes))

			# Figure other particles involed in this permutation
			# Meanwhile obtain the current state of permutation of these particles
			involved_particles = []
			current_array = []
			for ii in indexes:
				tmp0 = self.particles[ii].getOwnBead(k)
				while 1:
					if tmp0.belongsto not in involved_particles:
						involved_particles.append(tmp0.belongsto)
					tmp = tmp0.next
					if tmp.belongsto != tmp0.belongsto and tmp.belongsto not in current_array:
						current_array.append(tmp.belongsto)
					if tmp is self.particles[ii].getOwnBead(k):
						if tmp.belongsto not in current_array:
							current_array.append(tmp.belongsto)
						break
					tmp0 = tmp
			logger.info('involved_particles: %s' % str(involved_particles))
			logger.info('current_array:      %s' % str(current_array))
			# get new state of permutation
			new_array = []
			for ii in current_array:
				if ii not in indexes:
					new_array.append(ii)
				else:
					new_array.append(perm_indexes[indexes.index(ii)])
			# figure out real operators
			indexes0 = []
			perm_indexes0 = []
			for ii in range(len(current_array)):
				if current_array[ii] == new_array[ii]:
					continue
				else:
					indexes0.append(involved_particles[ii])
					perm_indexes0.append(new_array[ii])

			logger.info('new_array:          %s' % str(new_array))
			logger.info('real operator:      %s' % str(indexes0))
			logger.info('                    %s' % str(perm_indexes0))

			# record old actions
			old_action = 0.0
			for j in range(s-1):
				old_action += self.__PotentialAction(k+j+1)

			old_connection = {}
			old_position = {}
			old_shifts = {}
			# Do permutation:

			for ii in range(len(indexes0)):
				ii1 = indexes0[ii]
				ii2 = perm_indexes0[ii]
				r1 = self.particles[ii1].getOwnBead(k+s-1).r
				r2 = self.particles[ii2].getOwnBead(k+s).r
				# Record old information
				old_connection[ii1] = deepcopy(self.particles[ii1].getOwnBead(k+s-1).next.belongsto)
				old_position[ii1] = [np.zeros(d,dtype=float)]*s
				old_shifts[ii1] = [np.zeros(d,dtype=int)]*s
				for gg in range(s):
					old_position[ii1][gg] = deepcopy(self.particles[ii1].getOwnBead(k+gg).r)
					old_shifts[ii1][gg] = deepcopy(self.particles[ii1].getOwnBead(k+gg).nextShift)

				period_shift = np.zeros(3, dtype=int)
				for dd in range(d):
					if self.__NearerOppsite(ii1, ii2, k, s, dd):
						if r1[dd] <= r2[dd]:
							period_shift[dd] = -1
						else:
							period_shift[dd] = 1
				# CORE
				self.__connect(self.particles[ii1].getOwnBead(k+s-1), self.particles[ii2].getOwnBead(k+s), period_shift = period_shift) 
				self.BisectionMove(self.particles[ii1].getOwnBead(k), self.particles[ii2].getOwnBead(k+s), onlyMove = True)

			# calculate new_action
			new_action = 0.0
			for j in range(s-1):
				new_action += self.__PotentialAction(k+j+1)

			if np.random.random() < np.exp(-(new_action - old_action)):
				# Accepted!
				logger.info('Permutation done')
				return True
			else:
				# Rejected!
				logger.info('BisectionMove Rejected!')					
				logger.info('restoring...')

				# restore to the previous state:
				for ii in range(len(indexes0)):
					ii1 = indexes0[ii]
					# restore the connection
					self.particles[ii1].getOwnBead(k+s-1).next = self.particles[ old_connection[ii1] ].getOwnBead(k+s)
					self.particles[old_connection[ii1]].getOwnBead(k+s).prev = self.particles[ii1].getOwnBead(k+s-1)
					# restore the positions and period shifts
					for gg in range(s):
						self.particles[ii1].getOwnBead(k+gg).r = old_position[ii1][gg]
						self.particles[ii1].getOwnBead(k+gg).nextShift = old_shifts[ii1][gg]

				return False
		else:
			logger.info('Permutation Rejected! None exchanged.')
			return False

# ================================================================================================
def timeLeft(seconds):
	H = int(seconds/3600.)
	M = int((seconds % 3600)/60)
	S = int((seconds % 3600) % 60)
	return '%d:%d:%d' % (H, M, S)

def WindingNumberEstimator(cfg):
	'''
	Counts out winding number for the cfg configuration at current state.
	'''
	traversed_ptcl_indexes = []
	WindingNumber = np.array([0,0,0])
	for i in range(cfg.numParticles):
		if i in traversed_ptcl_indexes: 
			continue
		else:
			tmp0 = cfg.particles[i].origin
			traversed_ptcl_indexes.append(tmp0.belongsto)
			start = tmp0
			# method 1
			# while 1:
			# 	WindingNumber += tmp0.nextShift
			# 	tmp0 = tmp0.next
			# 	if tmp0 is start: break
			# if not (WindingNumber == np.array([0,0,0])).all():
			# 	a = raw_input('wrong @ $%d particle!' % i)

			# method 2
			while 1:
				tmp = tmp0.next
				if tmp.belongsto not in traversed_ptcl_indexes:
					traversed_ptcl_indexes.append(tmp.belongsto)
				deltshift = tmp0.nextShift

				for dd in range(d):
					# 2 sides +
					if (tmp0.r[dd] < 0.5*L[dd]) and (tmp.r[dd] >= 0.5*L[dd]):
						WindingNumber[dd] += deltshift[dd] + 1
					# 2 sides -
					elif (tmp0.r[dd] >= 0.5*L[dd]) and (tmp.r[dd] < 0.5*L[dd]):
						WindingNumber[dd] += deltshift[dd] - 1
					# 1 side
					else:
						WindingNumber[dd] += deltshift[dd]

				if tmp is start:
					break
				tmp0 = tmp
	# winding number is a vector here
	return WindingNumber


def MentoCarloSteps(cfg, MCSteps, ThermalSteps, TIndex, numTpoints, observeSkip):
	'''
	Does Mento Callo steps. 
	Continuously samples the paths spacial configuration and permutations.
	Oberves the winding number every other 'ObserveSkip'. 
	'''

	AcceptanceRatio = {'COM':0, 'Bisection':0, 'Permutation': 0}
	WindingNumberTrace = []
	etaSkip = 10
	# k, s defie the imaginary time window for permutation & bisection
	s = 4 
	k = 2
	dt = 0.
	eta = 'N/A'
	for step in range(MCSteps):
		t0 = time.time()
		logger.info('========Current step: %d/%d @T = %.2fK #%d/%d=========' \
	        % (step+1, MCSteps, cfg.Temperature*1.85529, TIndex, numTpoints))

		logger.info('Doing center of mass move...')
		for i in np.random.randint(0, cfg.numParticles, cfg.numParticles):
			start = cfg.particles[i].origin
			if cfg.ComMove(start):
				AcceptanceRatio['COM'] += 1

		logger.info('Doing bisection move...')
		for i in np.random.randint(0, cfg.numParticles, cfg.numParticles):
			[length] = random.sample([2,4,8,16], 1)
			alpha = np.random.randint(0, cfg.numTimeSlices-1-length)
			if cfg.BisectionMove(cfg.particles[i].getOwnBead(alpha), cfg.particles[i].getOwnBead(alpha + length)):
				AcceptanceRatio['Bisection'] += 1

		logger.info('Doing permutation-bisection move...')
		[max_sample_particles] = random.sample([2,3,4,5,6,7,8], 1)
		# [s] = random.sample([2,4,8,16], 1)
		# k = np.random.randint(0, cfg.numTimeSlices-1-length)
		if cfg.perm_bisecMove(k, s, max_sample_particles):
			AcceptanceRatio['Permutation'] +=1

		if (step+1) % observeSkip == 0 and step > ThermalSteps:
			logger.info('Observing...')
			W = WindingNumberEstimator(cfg)
			WindingNumberTrace.append(W)
			logger.info('--------------------------------------------------')
			logger.info('Result W = %s' % str(W))
			logger.info('--------------------------------------------------')
		dt += time.time()-t0
		if (step+1) % etaSkip == 0:
			eta = timeLeft(dt*(MCSteps-step-1)/etaSkip + dt*MCSteps*(numTpoints-TIndex)/etaSkip)
			dt = 0.
		logger.info('eta = %s' % eta)
	logger.info('---------------------Runtime Information---------------------')
	logger.info('Acceptance Ratios:')
	logger.info('Center of Mass: %.3f' % ((1.0*AcceptanceRatio['COM'])/(MCSteps*cfg.numParticles)))
	logger.info('Permutation:    %.3f' % ((1.0*AcceptanceRatio['Permutation'])/MCSteps))
	tot = 0.
	for W in WindingNumberTrace:
		tot += np.dot(W, W)
	meanSquare = tot/len(WindingNumberTrace)
	superfulid_density = meanSquare*cfg.Temperature*L[d-1]**2/(3*cfg.numParticles)

	# return superfulid_density
	return (meanSquare, superfulid_density, WindingNumberTrace)

def winding_number_dist_painter(winding_record, Temperature):
	sorted_dic= sorted(winding_record.iteritems(), key=lambda d:d[0])
	n_w = [x[1] for x in sorted_dic]
	groups = [int(x[0]) for x in sorted_dic]
	n_groups = len(groups)
	index = np.arange(n_groups)

	bar_width = 0.35

	fig, ax = plt.subplots()
	rects = plt.bar(groups, n_w, bar_width, alpha = 0.8, color = 'cornflowerblue')
	plt.xticks(np.array(groups)+0.5*bar_width, groups)
	plt.title('WindingNumber Distribution @ %.3f K' % Temperature)
	plt.xlabel('W')
	plt.ylabel('Num')
	plt.savefig('winding_number_%.3fK.png' % Temperature)


def main():
	'''
	The entrance of the program.
	'''
	DATA_FILE = 'data0.txt'
	# Temperature = [0.2, 0.5, 1.18, 2.0, 2.5] # unit in K
	Temperature = [1.18, 2.0]

	winding_number_dist = {}

	numTpoints = len(Temperature)
	f=open(DATA_FILE, 'w+r')
	if f.read() == '':
		f.write('numParticles = %d\n' % N)
		f.write('numTimeSlices = %d\n' % M)
		f.write('Mento Carlo Steps = %d\n' % MCSteps )
		f.write('T(K)\t<W2>\tDensity\tt(h)\tLocal_Time\n')
	f.close()
	TIndex = 0
	for T in Temperature:
		TIndex += 1
		logger.info('***********CURRENT TEMPERATURE: %.2f************ ' % T)
		t0 = time.time()
		cfg = configuration(numParticles = N, numTimeSlices = M, Temperature = T/1.85529, lam = 0.5)

		(W2, superfluid_density, WindingNumberTrace) = MentoCarloSteps(cfg, MCSteps=MCSteps, ThermalSteps = ThermalSteps, TIndex=TIndex, \
			numTpoints=numTpoints ,observeSkip=observeSkip)

		t_consumed = (time.time() - t0)/3600.
		logger.info('One Temperature point finished, result:')
		logger.info('--------------------------------------------------------------------------')
		logger.info('T(K)\t<W2>\tt(h)\tLocal_Time')
		localtime = time.strftime("%Y%m%d %H:%M:%S",time.localtime(time.time()))
		data = '%.3f\t%.3f\t%.3f\t%.3f\t'% ( T, W2, superfluid_density, t_consumed ) + localtime
		logger.info(data)
		logger.info('--------------------------------------------------------------------------')
		logger.info('Wrinting File...')
		# Calculate distribution of winding number
		winding_record = {}
		for W in WindingNumberTrace:
			if W[d-1] not in winding_record:
				winding_record[W[d-1]] = 1
			else:
				winding_record[W[d-1]] += 1

		# Paint the winding number distribution
		winding_number_dist_painter(winding_record, T)
		winding_number_dist[T] = winding_record
		picklefile = open('winding.pkl','wb')
		pickle.dump(winding_number_dist, picklefile)
		picklefile.close()

		f = open(DATA_FILE,'a')
		f.write(data+'\n')
		f.close()

		logger.info('File closed!\n')
	logger.info('All Done!')
	f = open(DATA_FILE)
	datas = f.read()
	logger.info('Data recorded:')
	logger.info('\n'+datas)


if __name__ == "__main__":
	main()





			



