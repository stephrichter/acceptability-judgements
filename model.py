import argparse, pandas, pymc
import numpy as np
import scipy as sp
import cPickle as pickle

#from waic import *


np.random.seed(1000)

def construct_prob_trace(trace, responses):
    return trace[:,range(responses.shape[0]),responses]

def compute_lppd(prob_trace):
    return np.log(prob_trace.mean(axis=0)).sum()

def compute_p_waic(prob_trace, method=2):
    if method == 1:
        mean_log = np.log(prob_trace).mean(axis=0)
        log_mean = np.log(prob_trace.mean(axis=0)) 
        
        return 2 * (log_mean - mean_log).sum() 
    elif method == 2:
        return np.log(prob_trace).var(axis=0).sum()
    else:
        raise ValueError('method parameter must be either 1 or 2'  )

def compute_waic(prob_trace, method=2):
    lppd = compute_lppd(prob_trace=prob_trace)
    p_waic = compute_p_waic(prob_trace=prob_trace, method=method)

    return -2 * (lppd - p_waic)

##################
## argument parser
##################

## initialize parser
parser = argparse.ArgumentParser(description='Load data and run violations model.')

## file handling
parser.add_argument('--data', 
                    type=str, 
                    default='../data/data.for.aaron.csv')
parser.add_argument('--outputdir', 
                    type=str, 
                    default='./model/')

## model specification
parser.add_argument('--unboundedviolation', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--violationtype', 
                    type=str, 
                    choices=['none', 'continuous', 'discrete'],
                    default='none')
parser.add_argument('--violationform', 
                    type=str, 
                    choices=['marginal', 'joint'], 
                    default='marginal')
parser.add_argument('--numofviolations', 
                    type=int, 
                    default=1)
parser.add_argument('--violationintercept', 
                    type=bool,
                    nargs='?', 
                    const=True,
                    default=False)
parser.add_argument('--additivesubjrandomeffects', 
                    nargs='?', 
                    const=True, 
                    default=False)
parser.add_argument('--multiplicativesubjrandomeffects', 
                    nargs='?', 
                    const=True, 
                    default=False)

## sampler parameters
parser.add_argument('--iterations', 
                    type=int, 
                    default=1100000)
parser.add_argument('--burnin', 
                    type=int, 
                    default=100000)
parser.add_argument('--thinning', 
                    type=int, 
                    default=1000)
parser.add_argument('--sampleonlyviolation', 
                    nargs='?', 
                    const=True, 
                    default=False)

## parse arguments
args = parser.parse_args()
    
#######################
## utility functions ##
#######################

def get_num_of_levels(series):
    '''Get the number of unique levels of a factor'''
    return series.cat.categories.shape[0]

###############
## load data ##
###############

data = pandas.read_csv(args.data)

data = data[data.distance != 'filler']

data.subject = data.subject.astype('category')
data.judgment = (data.judgment-1).astype('category', ordered=True)
data.item = data.item.astype('category')
data.dependency = data.dependency.astype('category')
data.island = data.island.astype('category')
data.structure = data.structure.astype('category', categories=['non', 'island'])
data.distance = data.distance.astype('category', categories=['short', 'long'])

"""BERT log probabilities"""
data.penalized_logprob = data.penalized_logprob.astype('float64')


##############################
## initialization functions ##
##############################

def init_did():
    try:
        return pickle.load(open('params/dependency_island_distance'))
    except IOError:
        print 'random initialization of dependency_island_distance'        
        return np.random.normal(0., 1.,
                                size=[get_num_of_levels(data.dependency),
                                      get_num_of_levels(data.island),
                                      get_num_of_levels(data.distance)])

def init_dis():
    
    try:
        return pickle.load(open('params/dependency_island_structure'))
    
    except IOError:
        print 'random initialization of dependency_island_structure'
        return np.random.normal(0., 1.,
                                size=[get_num_of_levels(data.dependency),
                                      get_num_of_levels(data.island),
                                      get_num_of_levels(data.structure)])

def init_violation_intercept():
    
    try:
        violation = pickle.load(open('params/violation'))
        return violation.min()
                    
    except IOError:
        print 'deterministic initialization of violation_intercept'
        return 0.

def init_violation_scale():

    try:
        violation = pickle.load(open('params/violation'))
        intercept = init_violation_intercept()
        violation_zeroed = violation - intercept
        return violation_zeroed.max()/args.numofviolations
        
    except IOError:
        print 'deterministic initialization of violation_scale'
        return 1.

    
def init_violation():
    try:
        violation = pickle.load(open('params/violation'))

        if args.unboundedviolation:
            return violation
        else:
            violation_scale = init_violation_scale()
            violation_zeroed = violation - init_violation_intercept()

            if args.violationtype == 'continuous':
                if args.violationintercept:
                    return violation_zeroed/violation_scale
                else:
                    return violation/violation_scale
                    
            else:
                if args.violationintercept:
                    return np.round(violation_zeroed/violation_scale)
                else:
                    return np.round(violation/violation_scale)
                            
    except IOError:
        print 'random initialization of violation'
        
        if args.unboundedviolation:
            return np.random.exponential(1.,
                                         size=[get_num_of_levels(data.dependency),
                                               get_num_of_levels(data.island)])

        elif args.violationtype == 'discrete':
            return np.random.binomial(n=args.numofviolations,
                                      p=.5,
                                      size=[get_num_of_levels(data.dependency),
                                            get_num_of_levels(data.island)])

        elif args.violationtype == 'continuous':
            return np.random.uniform(low=0.,
                                     high=args.numofviolations,
                                     size=[get_num_of_levels(data.dependency),
                                           get_num_of_levels(data.island)])

def init_item():
    try:
        return pickle.load(open('params/intercepts_item'))
    except IOError:
        print 'random initialization of intercepts_item'
        return np.random.normal(0., 1., size=get_num_of_levels(data.item))


def init_subjadd():
    try:
        return pickle.load(open('params/intercepts_subj_add'))
    except IOError:
        print 'random initialization of intercepts_subj_add'
        return np.random.normal(0., 1., size=get_num_of_levels(data.subject))

def init_subjmult():
    try:
        return pickle.load(open('params/intercepts_subj_mult'))
    except IOError:
        print 'random initialization of intercepts_subj_mult'
        return np.ones(get_num_of_levels(data.subject))
    
def init_jump():
    try:
        return pickle.load(open('params/jump'))
    except IOError:
        return sp.stats.expon.rvs(scale=.1,
                                  size=get_num_of_levels(data.judgment)-1)

                    
###################
## fixed effects ##
###################



dependency_island_distance = pymc.Normal(name='dependency_island_distance',
                                       mu=0.,
                                       tau=1e-6,
                                       value=init_did(),
                                       observed=False)

dependency_island_structure = pymc.Normal(name='dependency_island_structure',
                                          mu=0.,
                                          tau=1e-6,
                                          value=init_dis(),
                                          observed=False)

@pymc.deterministic
def fixed_tensor(did=dependency_island_distance, dis=dependency_island_structure):
    return did[:,:,:,None] + dis[:,:,None,:]


if args.unboundedviolation:

    violation = pymc.Exponential(name='violation',
                                 beta=1.,
                                 value=init_violation(),
                                 observed=False)

    @pymc.deterministic
    def fixed(fixed_tensor=fixed_tensor, violation=violation):
        no_violations = fixed_tensor[data.dependency.cat.codes,
                                     data.island.cat.codes,
                                     data.distance.cat.codes,
                                     data.structure.cat.codes]

        violations = violation[data.dependency.cat.codes,
                               data.island.cat.codes]

        return no_violations - data.distance.cat.codes * data.structure.cat.codes * violations

    
if args.violationtype == 'none':

    @pymc.deterministic
    def fixed(fixed_tensor=fixed_tensor):
        return fixed_tensor[data.dependency.cat.codes,
                            data.island.cat.codes,
                            data.distance.cat.codes,
                            data.structure.cat.codes]

else:

    violation_intercept = pymc.Exponential(name='violation_intercept',
                                       beta=1.,
                                       value=init_violation_intercept(),
                                       observed=False)
    
    violation_scale = pymc.Exponential(name='violation_scale',
                                       beta=1.*args.numofviolations,
                                       value=init_violation_scale(),
                                       observed=False)

    
    if args.violationform == 'joint':
        violation_propensity = pymc.Normal(name='violation_propensity',
                                           mu=0.,
                                           tau=1e-6,
                                           value=np.random.normal(0., 1.,
                                                                  size=[get_num_of_levels(data.dependency),
                                                                        get_num_of_levels(data.island),
                                                                        args.numofviolations]),
                                           observed=False)

        violation_prob = pymc.InvLogit(name='violation_prob', ltheta=violation_propensity)

        if args.violationtype == 'discrete':
            violation = pymc.Bernoulli(name='violation',
                                       p=violation_prob,
                                       observed=False)

        else:
            violation = violation_prob

    elif args.violationform == 'marginal':

        if args.violationtype == 'discrete':
            violation = pymc.DiscreteUniform(name='violation',
                                             lower=0,
                                             upper=args.numofviolations,
                                             value=init_violation(),
                                             observed=False)

        else:
            violation = pymc.DiscreteUniform(name='violation',
                                 lower=0,
                                 upper=args.numofviolations,
                                 value=init_violation(),
                                 observed=False)

            
        
    @pymc.deterministic
    def fixed(fixed_tensor=fixed_tensor, violation=violation, violation_intercept=violation_intercept, violation_scale=violation_scale):
        no_violations = fixed_tensor[data.dependency.cat.codes,
                                     data.island.cat.codes,
                                     data.distance.cat.codes,
                                     data.structure.cat.codes]

        if args.violationform == 'joint':
            violation_sum = violation.sum(axis=2)[data.dependency.cat.codes,
                                                  data.island.cat.codes]

        else:
            violation_sum = violation[data.dependency.cat.codes,
                                      data.island.cat.codes]

        if args.violationintercept:            
            return no_violations -\
              data.distance.cat.codes * data.structure.cat.codes * violation_intercept -\
              data.distance.cat.codes * data.structure.cat.codes * violation_scale * violation_sum
        else:
            return no_violations -\
              data.distance.cat.codes * data.structure.cat.codes * violation_scale * violation_sum
            
####################              
## random effects ##
####################

intercepts_item_prior = pymc.Gamma(name='intercepts_item_prior', 
                                   alpha=0.001,
                                   beta=1/0.001,
                                   value=sp.stats.expon.rvs(scale=.1),
                                   observed=False)

intercepts_item = pymc.Normal(name='intercepts_item',
                              mu=0.,
                              tau=intercepts_item_prior,
                              value=init_item(),
                              observed=False)



################
## likelihood ##
################

jump = pymc.Gamma(name='jump', 
                  alpha=0.001,
                  beta=1/0.001,
                  value=init_jump(),
                  observed=False)


"""Probability scores from BERT."""
prob_scoring_prior = pymc.Gamma(name='prob_scoring_prior', 
                                       alpha=0.001,
                                       beta=1/0.001,
                                       value=sp.stats.expon.rvs(scale=.1),
                                       observed=False)

prob_scoring = pymc.Normal('prob_scoring',
                                mu=0.,
                                tau=prob_scoring_prior,
                                #value=np.random.normal(0., 1., size=get_num_of_levels(data.penalized_logprob)),
                                observed=False)


if args.additivesubjrandomeffects:

    intercepts_subj_add_prior = pymc.Gamma(name='intercepts_subj_add_prior', 
                                       alpha=0.001,
                                       beta=1/0.001,
                                       value=sp.stats.expon.rvs(scale=.1),
                                       observed=False)
        
    intercepts_subj_add = pymc.Normal(name='intercepts_subj_add',
                                  mu=0.,
                                  tau=intercepts_subj_add_prior,
                                  value=init_subjadd(),
                                  observed=False)

    """Probability scoring added here as test"""
    @pymc.deterministic
    def param(fixed=fixed, intercepts_subj_add=intercepts_subj_add, intercepts_item=intercepts_item, prob_scoring=prob_scoring):
        return fixed + intercepts_subj_add[np.array(data.subject.cat.codes)] + intercepts_item[np.array(data.item.cat.codes)] + np.array(prob_scoring, dtype='float64')

else:

    @pymc.deterministic
    def param(fixed=fixed, intercepts_item=intercepts_item, prob_scoring=prob_scoring):
        return fixed + intercepts_item[np.array(data.item.cat.codes)] + np.array(prob_scoring, dtype='float64')

    
if args.multiplicativesubjrandomeffects:

    intercepts_subj_mult_prior = pymc.Gamma(name='intercepts_subj_mult_prior', 
                                       alpha=0.001,
                                       beta=1/0.001,
                                       value=sp.stats.expon.rvs(scale=.1),
                                       observed=False)
        
    intercepts_subj_mult = pymc.Exponential(name='intercepts_subj_mult',
                                       beta=intercepts_subj_mult_prior,
                                       value=init_subjmult(),
                                       observed=False)

    @pymc.deterministic
    def log_prob(jump=jump, param=param, intercepts_subj_mult=intercepts_subj_mult):
        jump_warped = (1/intercepts_subj_mult[data.subject.cat.codes,None])*jump

        return np.cumsum(jump_warped, axis=1)-param[:,None]

else:

    @pymc.deterministic
    def log_prob(jump=jump, param=param):
        return np.cumsum(jump)[None,:]-param[:,None]


@pymc.deterministic
def prob(log_prob=log_prob):
    cdfs = 1 / (1+np.exp(-log_prob))
    
    zeros = np.zeros(data.shape[0])[:,None]
    ones = np.ones(data.shape[0])[:,None]

    return np.append(cdfs, ones, axis=1) - np.append(zeros, cdfs, axis=1)


judgment = pymc.Categorical(name='judgment',
                            p=prob,
                            value=data.judgment,
                            observed=True)


def dump_vals():
    dependency_island_distance.value.dump('params/dependency_island_distance')
    dependency_island_structure.value.dump('params/dependency_island_structure')

    violation.value.dump('params/violation')

    jump.value.dump('params/jump')
    
    intercepts_item.value.dump('params/intercepts_item')
    
    if args.additivesubjrandomeffects:
        intercepts_subj_add.value.dump('params/intercepts_subj_add')

    if args.multiplicativesubjrandomeffects:
        intercepts_subj_mult.value.dump('params/intercepts_subj_mult')
                
###################
## model fitting ##
###################

if args.violationtype=='none' and not args.sampleonlyviolation:
    ## fit a MAP estimate 
    model = pymc.MAP(locals())
    model.fit(method='fmin_l_bfgs_b', iterlim=100000)
    
## initialize model and begin sampler
if args.sampleonlyviolation:
    if args.violationtype == 'continuous': 
        model = pymc.MCMC([violation, prob])
    else:
        model = pymc.MCMC([violation, violation_scale, prob])
else:
    model = pymc.MCMC(locals())
        
model.sample(iter=args.iterations, burn=args.burnin, thin=args.thinning)

if not args.unboundedviolation and not args.violationtype=='continuous' and not args.sampleonlyviolation:
    dump_vals()


#prob_trace = construct_prob_trace(prob.trace(),
#                                  data.judgment.astype(int))
#lppd = compute_lppd(prob_trace)
##waic = compute_waic(prob_trace)




with open('performance.csv', 'a') as f:
    line = '{},{},{},{},{},{}\n'.format(args.violationtype,
                                str(args.unboundedviolation),
                                str(args.violationintercept),
                                str(args.numofviolations),
                                model.DIC,
                                model.BPIC,
                                #waic
                                )
                                # model.dic
    f.write(line)

# pandas.melt(pandas.Panel(violation.trace(), items=range(1000), major_axis=data.dependency.cat.categories, minor_axis=data.island.cat.categories).to_frame().reset_index(), id_vars=['major', 'minor'])
    
## get deviance trace, minimum deviance, and index of minimum deviance
# deviance_trace = model.trace('deviance')()
# deviance_min = deviance_trace.min()
# minimum_index = np.where(deviance_trace == deviance_min)[0][0]
