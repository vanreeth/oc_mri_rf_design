import numpy as np
from dataclasses import dataclass, field

@dataclass
class Global:
    """
    Represents global parameters for an application. This class contains attributes to define 
    the control time duration, targetd nucleus and number of time points.
    These parameters will be unchanged during the optimal control process.

    Attributes:
        application (str): Specifies the application type. Default is 'selective_excitation'.
            Must be one of: 'selective_excitation', 'b1_robust_excitation', 'mre'
        N (int): The number of time steps. Default is 256. Must be a power of 2 for wavelet parametrization.
        T_s (float): The total control time duration in seconds. Default is 5 ms.
        nucleus (str): The targeted nucleus for the excitation. Default is 'H' (Hydrogen). 
            Must be one of: 'H', 'Na', 'C', 'P' 
    """
    application: str = 'selective_excitation'
    N: int = 256  # number of time steps
    T_s: float = 5e-3  # control time duration in seconds
    nucleus: str = 'H'  # targeted nucleus  
    refoc_f: float = 0.5 # factor of T_s to use in refoc

    def __post_init__(self):
        self.t_s = np.arange(self.N)/self.N * self.T_s # time vector (in s) associated with the rf pulse
        "Check if the application type is valid"
        valid_applications = ['selective_excitation', 'b1_robust_excitation', 'mre']
        if self.application not in valid_applications:
            raise ValueError(f"Invalid application: {self.application}. "
                             f"Must be one of {valid_applications}")
        valid_nucleus = ['H', 'P', 'Na', 'C']
        if self.nucleus not in valid_nucleus:
            raise ValueError(f"Invalid choice of nucleus: {self.nucleus}. "
                             f"Must be one of {valid_nucleus}")
        else:
            self.gamma = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}.get(self.nucleus, 267.513e6)

              
@dataclass
class Population:
    """
    Represents a population with specific physical properties.
    
    Attributes:
        application : str
            The application type. Default is 'selective_excitation'.
        T1 : float
            T1 relaxation time in seconds. Default is infinity (`np.inf`).
        T2 : float
            T2 relaxation time in seconds. Default is infinity (`np.inf`).
        PD : float
            Proton density (normalized). Default is 1.
        M0 : float
            Magnetization at equilibrium (normalized). Default is 1.
        CS : float
            Chemical shift in radians per second. Default is 0.
        B1p : float
            B1-plus factor (normalized). Default is 1.
        motion_props : dict
            A dictionary specifying motion properties. Default is an empty dictionary.
        tissue : str
            The name of the tissue associated with these properties. Default is `None`.
        population_idx : int
            The index of the population, useful for visualizing and differentiating populations (e.g., in plots).
            Default is 0.
    """
    application: str    = 'selective_excitation' # application type
    T1:float            = np.inf # T1 relaxation time in seconds
    T2:float            = np.inf # T2 relaxation time in seconds
    PD:float            = 1 # Proton density (normalized)
    # M0:np.ndarray       = np.array([0,0,1]) # Magnetization at equilibrium (normalized)
    M0: np.ndarray = field(default_factory=lambda: np.array([0,0,1]).squeeze())# Magnetization at equilibrium (normalized)
    CS:float            = 0 # Chemical shift in rad/s
    B1p:float           = 1 # B1-plus factor (normalized)
    motion_props: dict  = field(default_factory=dict) # motion properties (e.g. diffusion, flow, mre)
    tissue: str         = None # Name of the tissue having these properties
    population_idx: int = 0 # Index of the population (useful for plots)
        
    def __post_init__(self):
        if self.application == 'mre':
            if not self.motion_props:
                self.motion_props = {'type':'sine', 'motion_amplitude_mm':40e-3, 'motion_frequency_hz':400, 'motion_phase':0}       

@dataclass
class Geometry:
    """
    Configuration of the problem geometry depending on the specified application.

    Attributes:
        application (str): Specifies the application type to set up default values. 
            Default is 'selective_exc'. Must be one of: 'selective_excitation', 'b1_robust_excitation'
        sliceThickness_mm (float): The thickness of the slice in millimeters.
        stepSlice_mm (float): The spatial step size inside the slice, measured in millimeters.
        transitionWidth_mm (float): The width of the transition band in millimeters
        nbTransBand (int): The number of isochromats in the positive transition band, 
            used to discretize the transition region.
        maxStopBand_mm (float): The location of the last controlled isochromat in the 
            stop band, measured in millimeters.
        nbStopBand (int): The number of isochromats in the positive stop band, 
            determining the level of spatial resolution in this region.
        logStep (bool): If True, uses a logarithmic distribution for the positions 
            of isochromats, offering finer control at positions close to 0
        halfPos (bool): If True, considers only positive positions for isochromat 
            distribution, applicable to w1x optimizations.
    """
    application: str            = 'selective_excitation' # application type
    sliceThickness_mm: float    = 1.0   # slice thickness (mm)
    nb_in_slice: int            = 25  # spatial step inside the slice (mm)
    transitionWidth_mm: float   = 0.15  # width of the transition band
    nb_transBand: int           = 0     # Nb of isochromats in the positive transition band
    max_stopBand_mm: float      = 5.0   # location of the last controlled isochromat
    nb_stopBand: int            = 200   # Nb of isochromats in the positive stopband
    logStep: bool               = True  # log distribution of the isochromats position   
    halfPos: bool               = True  # consider only positive position in case of w1x optimization only           
                
    
    # if application == 'selective_excitation':
        # sliceThickness_mm: float = 1.0  # slice thickness (mm)
        # stepSlice_mm: float = 0.01       # spatial step inside the slice (mm)
        # transitionWidth_mm: float = 0.15 # width of the transition band
        # nbTransBand: int = 0           # Nb of isochromats in the positive transition band
        # maxStopBand_mm: float = 5.0      # location of the last controlled isochromat
        # nbStopBand: int = 200           # Nb of isochromats in the positive stopband
        # logStep: bool = True            # log distribution of the isochromats position   
        # halfPos: bool = True           # consider only positive position in case of w1x optimization only
    # elif application == 'b1_robust_excitation':
    #     sliceThickness_mm: float = 0  # slice thickness (mm)
    #     stepSlice_mm: float = 0       # spatial step inside the slice (mm)
    #     transitionWidth_mm: float = 0 # width of the transition band
    #     nbTransBand: int = 0           # Nb of isochromats in the positive transition band
    #     maxStopBand_mm: float = 0      # location of the last controlled isochromat
    #     nbStopBand: int = 0           # Nb of isochromats in the positive stopband
    #     logStep: bool = False            # log distribution of the isochromats position   
    #     halfPos: bool = False           # consider only positive position in case of w1x optimization only
    
    def __post_init__(self):
        sT      = self.sliceThickness_mm
        N_in    = self.nb_in_slice
        N_out   = self.nb_stopBand
        a=np.linspace(-sT/2, sT/2, N_in)
        p=np.unique(np.concatenate(([0],a)).round(decimals=10)) #Problema con doble 0
        if self.halfPos:
            self.Nin  =len(p[p>=0])
            self.Nout = N_out//2
        else:
            self.Nin  = len(p)
            self.Nout = N_out
        
@dataclass
class Algo:
    """
    Configuration for algorithm parameters and output management.

    Attributes:
        max_iter (int): Maximum number of iterations for the algorithm. 
            Default is 1000.
        export_path (str): Path where output data will be saved. 
            Default is './'.
        gradient_tolerance (float): Tolerance level for the gradient to determine 
            convergence. The algorithm stops when the gradient falls below this threshold. 
            Default is 1e-7.
    """
    max_iter: int = 1000 # number of maximum iterations
    export_path: str = './' # export path to save output data
    gradient_tolerance: float = 1e-7 # gradient tolerance for convergence stopping criteria
    function_tolerance: float = 1e-7 # gradient tolerance for convergence stopping criteria
    tolerance: float = 1e-6 # general tolerance stopping criteria
    method: str = 'trust-constr'
    sol_folder: str ='last_optimization'

@dataclass
class Cost:
    """
    Defines the cost function and some parameters used to define the optimal control problem. 
    This class initializes target values, and weights required to set-up the individual isochromat
    target depending on their properties.

    Attributes:
        application (str): Specifies the application type to set up default values. 
            Default is 'selective_excitation'. Must be one of: 'selective_excitation', 'b1_robust_excitation'
        cost_function (str): The cost function to be used for optimization.
            Must be one of 'distance_to_target','feasibility','min_energy','min_max_amplitude'
        target_flipAngle (dict): A dictionary specifying target flip angles (in radians) 
            based on some valid isochromat attributes:
            target_flipAngle = {
                'criterion':'slice_status', # criterion to determine target flip angles (must be a valid isochromat attribute)
                'names':['in','out','trans'], # fields taken by the specified criterion
                'values':[np.pi/2, 0, 0] # target flip angles (rd) associated with the fields previously defined in 'names'
            }
        target_phase (dict): A dictionary specifying target phases (in radians) based on 
            isochromat attributes:
            target_phase = {
                'static_phase':0, # static phase (rd) target applied to all isochromats
                'linear':True, # Boolean to specify if a linear phase distribution is considered
                'phaseFactor':-0.5, # phase factor in proportion of the pulse duration
                'criterion':None, # criterion to determine target phases (must be a valid isochromat attribute)
                'names':None, # fields taken by the specified criterion
                'values':None # target phases in rd associated with the fields previously defined in 'names'
            }
        weights (dict): A dictionary specifying weights for different isochromat attributes:
            weights = {
                'criterion':'slice_status', # criterion to determine weights (must be a valid isochromat attribute)
                'names':['in','out','trans'], # fields taken by the specified criterion
                'values':[1,100,50] # weights associated with the fields previously defined in 'names'
            }
    """
    application: str          = 'selective_excitation'# Specify application to set-up default values
    cost_function: str        = 'distance_to_target' # Cost function to be used
    target_flipAngle: dict    = field(default_factory=dict) 
    target_phase: dict        = field(default_factory=dict) 
    weights: dict             = field(default_factory=dict) 
    def __post_init__(self):    
        valid_costs = ['distance_to_target','feasibility','min_energy','min_max_amplitude']
        if self.cost_function not in valid_costs:
            raise ValueError(f"Invalid cost function: {self.cost_function}. "
                             f"Must be one of {valid_costs}")
        if self.application == 'selective_excitation':
            if not self.cost_function:  # Check if the dictionary is empty
                self.cost_function: str = 'distance_to_target'
            if not self.target_flipAngle:  # Check if the dictionary is empty
                self.target_flipAngle: dict = {    # Dictionary to specify the target flip angles (in rd) depending on some isochromat attributes
                    'criterion':'slice_status',
                    'names':['in','out','trans'],
                    'values':[np.pi/2, 0, 0] # target flip angles (rd)       
                }
            if not self.target_phase:  # Check if the dictionary is empty
                self.target_phase: dict = {    # Dictionary to specify the target phases (in rd) depending on some isochromat attributes
                    'static_phase':0,
                    'linear':True,
                    'phaseFactor':-0.5, 
                    'criterion':None,
                    'names':None, # vector of values taken by criterion
                    'values':None # vector of target phases in rd associated with the filed previously defined in 'names'
                }
            if not self.weights:  # Check if the dictionary is empty
                self.weights: dict = {     # Dictionary to specify the weights depending on some isochromat attributes
                    'criterion':'slice_status',
                    'names':['in','out','trans'],
                    'values':[1,10,5] # weights
                }
    