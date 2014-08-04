from future.builtins import object
import abc
from klearn.kernels import AbstractKernel
from future.utils import with_metaclass

class BaseLearner(with_metaclass(abc.ABCMeta, object)):
    """
    Basic learner that implements what we want from the kernel 
    learners
    """

    def __init__(self, kernel):
        
        if not isinstance(kernel, AbstractKernel):
            raise Exception("kernel must be of type AbstractKernel")

        self.kernel = kernel


    @abc.abstractmethod
    def add_training_data(self, X):
        """
        Add new training data to the learner. It is up to the
        subclass to decide what to do with the data, as it may
        be advantageous to calculate something on the fly, or 
        just store it until later.
        """

        raise NotImplementedError("not implemented!")

    
    @abc.abstractmethod
    def solve(self):
        """
        Solve the learning method given all of the training 
        data. 
        """
        
        raise NotImplementedError("not implemented!")

    
    @abc.abstractmethod
    def save(self, filename):
        """
        save the results of the learning algorithm to a file
        """
        
        raise NotImplementedError("not implemented!")


    @classmethod
    def load(cls, filename):
        """
        load the results from a file saved using the cls.save 
        method
        """
        
        raise NotImplementedError("not implemented!")


class CrossValidatingMixin(with_metaclass(abc.ABCMeta, object)):
    """
    mixin for learners that can be cross-validated
    """

    @abc.abstractmethod
    def evaluate(self, X):
        """
        cross-validate new data given the solution
        """

        raise NotImplementedError("not implemented!")


class ProjectingMixin(with_metaclass(abc.ABCMeta, object)):
    """
    mixin for learners that project new data onto some
    solutions. I.e. these learners will transform the 
    data into one or more dimensions
    """

    @abc.abstractmethod
    def project(self, X, num_vecs=10):
        """
        project data, X, onto some number of solutions
        """

        raise NotImplementedError("not implemented!")
    
