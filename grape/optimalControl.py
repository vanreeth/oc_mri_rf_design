import numpy as np
from scipy.linalg import expm
from numba import jit
import plotly.express as px
from scipy.optimize import minimize
from scipy.optimize import check_grad
from cyipopt import minimize_ipopt
import copy
from joblib import Parallel, delayed
from pandas import DataFrame
from cyipopt import minimize_ipopt
from grape import propagation, Analysis
import numbers
import os


class OC:
    # Define the optimal control problem parameters 
    __epsilon = 1e-10
    __w1x_mat = np.array( [ [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 1.],
                            [0., 0., -1., 0.] ])
    __w1y_mat = np.array( [ [0., 0., 0., 0.], 
                            [0., 0., 0., -1.],
                            [0., 0., 0., 0.],
                            [0., 1., 0., 0.] ])
    __w0_mat = np.array( [ [0., 0., 0., 0.], 
                            [0., 0., 1., 0.],
                            [0., -1., 0., 0.],
                            [0., 0, 0., 0.] ])    

    def __init__(self, optim_parameters, population_list, control_fields):
        self.__application              = optim_parameters['global'].application
        self.__N                        = optim_parameters['global'].N
        self.__t_s                      = optim_parameters['global'].t_s
        self.__T_s                      = optim_parameters['global'].T_s
        self.__refoc_f                  = optim_parameters['global'].refoc_f
        # self.__t_m_s                    = np.arange(self.__N+1)/self.__N*optim_parameters['global'].T_s not used, remove?
        self.__dt                       = self.__t_s[1] - self.__t_s[0]
        self.__gamma                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}.get(optim_parameters['global'].nucleus, 267.513e6)
        self.__sliceThickness_mm        = optim_parameters['geometry'].sliceThickness_mm
        self.__transitionWidth_mm       = optim_parameters['geometry'].transitionWidth_mm
        self.Nin                        = optim_parameters['geometry'].Nin
        self.Nout                       = optim_parameters['geometry'].Nout
        self.__costParameters           = optim_parameters['cost']
        self.__maxIter                  = optim_parameters['algo'].max_iter
        self.__optim_method             = optim_parameters['algo'].method
        self.__g_tol                    = optim_parameters['algo'].gradient_tolerance
        self.__f_tol                    = optim_parameters['algo'].function_tolerance
        self.__tolerance                = optim_parameters['algo'].tolerance
        self.__constraints              = optim_parameters['constraints']
        self.__global                   = optim_parameters['global']
        self.__sol_folder               = optim_parameters['algo'].sol_folder

        ######## Set-up optimization problem
        # Compute isochromat positions 
        self.__pos_mm = self.compute_positions(optim_parameters)
       
        # Set-up isochromat list (isoList)
        self.isoList = self.generate_isoList(population_list, control_fields)
       
        # Analysis.plot_targetStates(self.isoList)
        
        # Propagate 
        #self.propagate_par(control_fields)
        self.isoList = propagation.propagate_par(control_fields, self.isoList ,self.__global) # __isoList
        
        # Set-up initial optimization variable (X0)
        X0 = []
        idx_X = {'w1x':[], 'w1y':[], 'gz':[]}
        idx_start = 0
        if control_fields['w1x'].optimize:
            X0.append(control_fields['w1x'].controlField)
            #X0.append(1000*np.ones(len(control_fields['w1x'].controlField)))
            idx_X['w1x'] = [idx_start, idx_start + control_fields['w1x'].controlField.size]
            idx_start += control_fields['w1x'].controlField.size
        if control_fields['w1y'].optimize:
            X0.append(control_fields['w1y'].controlField)
            idx_X['w1y'] = [idx_start, idx_start + control_fields['w1y'].controlField.size]
            idx_start += control_fields['w1y'].controlField.size
        if control_fields['gz'].optimize:
            X0.append(control_fields['gz'].controlField)
            idx_X['gz'] = [idx_start, idx_start + control_fields['gz'].controlField.size]
            idx_start += control_fields['gz'].controlField.size
        if self.__costParameters.cost_function=='min_max_amplitude':
            z=max(abs(control_fields['w1x'].scale*control_fields['w1x'].compute_tempShape())) #MODIFY INCLUDING w1y
            X0.append([z**2 / (self.__gamma * 1e-6)**2])
        self.__idx_X = idx_X
        self.__X0 = np.concatenate(X0)
        
    # Define getters & setters
    @property # to define getters
    def isoList(self):
        return self.__isoList #__isoList
    
    @isoList.setter
    def isoList(self, isoList):
        """Setter for the isoList private attribute"""
        self.__isoList = isoList #__isoList

    # Define methods
                                    
    def generate_isoList(self, population_list, control_fields):
        
        Constraints_names=['max_energy','Mxy_inSlice_lb', 'Mz_inSlice_ub','Mxy_lb','Mxy_outSlice_ub','max_peak_amplitude_uT','signal_in','max_amplitude_g']
        Constraints_used =self.__constraints.keys()
        if not set(set(Constraints_used)).issubset(set(Constraints_names)):
            raise TypeError('There are constrains with incorrect name, the valid names are:',  Constraints_names)
        

        isoList = []
        for h, pos in enumerate(self.__pos_mm):
            for i,p in enumerate (population_list):
                for j,t1 in enumerate (np.atleast_1d(p.T1).astype(np.float32)):
                    for k,t2 in enumerate (np.atleast_1d(p.T2).astype(np.float32)):
                        for l,pd in enumerate (np.atleast_1d(p.PD).astype(np.float32)):
                            for m,m0 in enumerate (np.atleast_3d(p.M0).astype(np.float32)):
                                for n,cs in enumerate (np.atleast_1d(p.CS).astype(np.float32)):
                                    for o,b1p in enumerate (np.atleast_1d(p.B1p).astype(np.float32)):   
                                        # Create isochromat object
                                        iso = {
                                            'T1':t1, 
                                            'T2':t2, 
                                            'PD':pd, 
                                            'M0':m0.squeeze(), 
                                            'CS':cs, 
                                            'B1p':b1p, 
                                            'pos_mm':pos, 
                                            'tissue':p.tissue, 
                                            'population_idx':p.population_idx,
                                            'motion_props':p.motion_props,
                                            }
                                            
                                        # Set slice status
                                        if np.abs(pos) <= round((self.__sliceThickness_mm)/2,5):
                                            iso['slice_status']    = 'in'
                                        elif np.abs(pos) < round((self.__sliceThickness_mm)/2 + self.__transitionWidth_mm,5):
                                            iso['slice_status']    = 'trans'
                                        else:
                                            iso['slice_status']    = 'out'                                          

                                        
                                        # Set target states if required by cost function
                                        if self.__costParameters.cost_function == 'distance_to_target':
                                            target_flipAngle = self.__costParameters.target_flipAngle
                                            for t in np.arange(len(target_flipAngle['names'])):
                                                if iso[target_flipAngle['criterion']] == target_flipAngle['names'][t]:
                                                    flip_angle = target_flipAngle['values'][t]

                                            target_phase = self.__costParameters.target_phase
                                            if target_phase['linear']:
                                                gz_temp  = control_fields['gz'].compute_tempShape()
                                                w0      = iso['CS'] + (iso['pos_mm'])*1e-3*self.__gamma*gz_temp[0]
                                                phi     = w0 * target_phase['phaseFactor'] * self.__T_s + np.pi/2
                                            elif target_phase['criterion'] != None:
                                                for t in np.arange(len(target_phase['names'])):
                                                    if iso[target_phase['criterion']] == target_phase['names'][t]:
                                                        phi = target_phase['values'][t]
                                            else:
                                                phi = 0
                                                        
                                            
                                            iso['target'] = np.array([np.sin(flip_angle)*np.cos(phi+target_phase['static_phase']), np.sin(flip_angle)*np.sin(phi+target_phase['static_phase']), np.cos(flip_angle)])
                                        else:
                                        #elif self.__costParameters.cost_function == 'feasibility':
                                            iso['target'] = None
                                        
                                        
                                        # Set constraints
                                        if 'Mxy_lb' in Constraints_used:
                                            iso['Mxy_lb'] = self.__constraints['Mxy_lb']['bound']
                                        
                                        if 'Mz_inSlice_ub' in Constraints_used:
                                            iso['Mz_inSlice_ub'] = self.__constraints['Mz_inSlice_ub']['bound']

                                        if 'Mxy_outSlice_ub' in Constraints_used:
                                            iso['Mxy_outSlice_ub'] = self.__constraints['Mxy_outSlice_ub']['bound']
                                            
                                        # Set weights
                                        weights = self.__costParameters.weights
                                        if weights:
                                            for t in np.arange(len(weights['names'])):
                                                if iso[weights['criterion']] == weights['names'][t]:
                                                    w = weights['values'][t]  
                                            iso['weight'] = w
                                        
                                        # Set motion if application is MRE
                                        if self.__application == 'mre':
                                            if p.motion_props['type'] == 'sine':
                                                m = p.motion_props['motion_amplitude_mm'] * np.sin(2*np.pi*p.motion_props['motion_frequency_hz']*np.arange(self.__N)*self.__dt + p.motion_props['motion_phase'])
                                            elif p.motion_props['type'] == 'linear':
                                                m = np.linspace(-1, 1, self.__N) * p.motion_props['amplitude_mm']/2
                                            else:
                                                m = np.zeros(self.__N)
                                            iso['motion_props']['motion'] = m
                                        
                                        isoList.append(iso)
        # self.isoList = isoList
        return isoList     
                                    
    def compute_positions(self,optim_parameters):
        # Set-up postion vector
        tW      = optim_parameters['geometry'].transitionWidth_mm
        sT      = optim_parameters['geometry'].sliceThickness_mm
        maxZ    = optim_parameters['geometry'].max_stopBand_mm
        nb_TB   = optim_parameters['geometry'].nb_transBand
        nb_SB   = optim_parameters['geometry'].nb_stopBand
        N_in    = optim_parameters['geometry'].nb_in_slice
        if optim_parameters['geometry'].logStep:
            p = np.unique(np.concatenate((
                [0],
                np.linspace(-sT/2, sT/2, N_in),  
                -np.linspace(sT/2, sT/2 + tW, nb_TB//2),
                np.linspace(sT/2, sT/2 + tW, nb_TB//2),
                -np.logspace(np.log10(tW + sT/2), np.log10(maxZ), nb_SB//2),
                np.logspace(np.log10(tW + sT/2), np.log10(maxZ), nb_SB//2)
            )).round(decimals=10))
        else:
            p = np.unique(np.concatenate((
                [0],
                np.linspace(-sT/2, sT/2, N_in),
                -np.linspace(sT/2, sT/2 + tW, nb_TB//2),
                np.linspace(sT/2, sT/2 + tW, nb_TB//2),
                -np.linspace(tW + sT/2, maxZ, nb_SB//2),
                np.linspace(tW + sT/2, maxZ, nb_SB//2)
            )).round(decimals=10))

        # Apply the halfPos logic
        if optim_parameters['geometry'].halfPos:
            p = p[p >= 0]
            
        return p

   
    def get_controlFields_from_X(self, X, control_fields):
        idx_X = self.__idx_X
        updated_cF = copy.deepcopy(control_fields)
        if idx_X['w1x']: # list not empty
            updated_cF['w1x'].controlField = X[ idx_X['w1x'][0] : idx_X['w1x'][1]]
        if idx_X['w1y']: # list not empty
            updated_cF['w1y'].controlField = X[ idx_X['w1y'][0] : idx_X['w1y'][1]]
        if idx_X['gz']: # list not empty
            updated_cF['gz'].controlField = X[ idx_X['gz'][0] : idx_X['gz'][1]]
        return updated_cF
    
    def run_optimization(self, control_fields):
        


        # Define constraints with Jacobians
        Constraints_used =self.__constraints.keys()    
        constraints = []
        if 'max_energy' in Constraints_used:
            constraints.append(
                {'type': 'ineq', 'fun': self.constraint_pulse_energy, 'jac': self.constraint_gradient_pulse_energy, 'args':(control_fields,)}
            )
        if ('Mxy_inSlice_lb' in Constraints_used) or ('Mxy_lb' in Constraints_used) or ('Mxy_outSlice_ub' in Constraints_used) or ('signal_in' in Constraints_used) or ('Mz_inSlice_ub' in Constraints_used):
            if 'Mxy_inSlice_lb' in Constraints_used and 'isoList_comp' in self.__constraints['Mxy_inSlice_lb'].keys():
                print('Mxy_inSlice_lb: bounds are set by isoList_comp, any other bounds will be overwritten.')
            if 'Mxy_outSlice_ub' in Constraints_used and 'isoList_comp' in self.__constraints['Mxy_outSlice_ub'].keys():
                print('Mxy_outSlice_ub: bounds are set by isoList_comp, any other bounds will be overwritten.')   
            if 'signal_in' in Constraints_used and 'isoList_comp' in self.__constraints['signal_in'].keys():
                print('signal_lb: bounds are set by isoList_comp, any other bounds will be overwritten.')   
                print('signal_lb can not be used when gz is being optimized.') 
            if 'Mz_inSlice_ub' in Constraints_used and 'isoList_comp' in self.__constraints['Mz_inSlice_ub'].keys():
                print('Mz_inSlice_ub: bounds are set by isoList_comp, any other bounds will be overwritten.')
            if 'Mxy_lb' in Constraints_used and self.isoList is None:
                stop=1
            else:
             constraints.append(
                {'type': 'ineq', 'fun': self.constraints_with_propagation, 'jac':True, 'args':(control_fields,)})
                 
        if 'max_peak_amplitude_uT' in Constraints_used:
            constraints.append(
                {'type': 'ineq', 'fun': self.constraint_max_peak_amplitude_uT, 'jac': self.constraint_gradient_max_peak_amplitude_uT, 'args':(control_fields,)}
            )
        if 'max_amplitude_g' in Constraints_used:
            constraints.append(
                {'type': 'ineq', 'fun': self.constraint_max_peak_amplitude_g, 'jac': self.constraint_gradient_max_peak_amplitude_g, 'args':(control_fields,)}
            )
        if self.__costParameters.cost_function=='min_max_amplitude':
            constraints.append(
                {'type': 'ineq', 'fun': self.constraint_min_max_peak_amplitude_uT, 'jac': self.constraint_gradient_min_max_peak_amplitude_uT, 'args':(control_fields,)}
            )    
        
        if self.__optim_method =='IPOPT':

            if not os.path.exists(self.__sol_folder):
                os.makedirs(self.__sol_folder)

            res=minimize_ipopt(
            x0=self.__X0, 
            args=(control_fields,),
            fun=self.compute_cost_and_grad,
            jac=True,
            constraints=constraints,
            tol=self.__tolerance,
            options={
                'max_iter': self.__maxIter,
                'print_level':5,
                'output_file':self.__sol_folder+'/sol_message.txt',
                #'nlp_scaling_method': 'gradient-based',
                'constr_viol_tol':5e-4,
                },  
            # callback=iteration_callback
            )
        else:
            res = minimize(
                x0=self.__X0, 
                args=(control_fields,), 
                method=self.__optim_method,
                fun=self.compute_cost_and_grad,
                jac=True,
                constraints=constraints,
                tol = self.__tolerance,
                options={
                    'maxiter': self.__maxIter,
                    # 'gtol': self.__g_tol, 
                    # 'ftol': self.__f_tol, 
                    'disp': True
                    }
                )
        optimized_control_fields = self.get_controlFields_from_X(res.x, control_fields=control_fields)
        self.isoList=propagation.propagate_par(optimized_control_fields, self.isoList ,self.__global)
        #self.isoList=propagation.propagation_rf(optimized_control_fields, self.isoList ,self.__global)
        return optimized_control_fields
    
    # Callback function to record each iteration's information
    # def callback_function(self, xk):
    #     # Save the current values of the parameters and objective function
    #     fval = objective(xk)
    #     iteration_data.append({"x": xk.copy(), "fval": fval})
  
  ########### Cost #############
    
    def compute_cost_and_grad(self, x, control_fields):
        updated_cF      = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        self.isoList = propagation.propagate_par(updated_cF, self.isoList, self.__global)
        #self.isoList = propagation.propagation_rf(updated_cF, self.isoList, self.__global)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape()
        gz_temp     = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape()

        # Compute cost
        match self.__costParameters.cost_function:
            case 'distance_to_target':
                # Compute cost
                individual_costs = Parallel(n_jobs=-1)( # Use all available cores
                    delayed(distance_to_target)(iso['M_at_T'], iso['target'], iso.get('weight', 1)) 
                    for iso in self.isoList
                    )
                cost = 1/len(self.isoList) * np.sum(individual_costs) 
                w0_mat      = self.__w0_mat
                w1x_mat     = self.__w1x_mat
                w1y_mat     = self.__w1y_mat
                optim_dict = {
                    'w1x':updated_cF['w1x'].optimize,
                    'w1y':updated_cF['w1y'].optimize,
                    'gz':updated_cF['gz'].optimize}
                # Get final adjoint state for all isochromats
                p_at_T = []
                for iso in self.isoList:
                    p_at_T.append( self.compute_final_adjoint_state(iso=iso) )
                
                # Compute gradient using parallel computation
                grad_dict_list = []
                grad_dict_list = Parallel(n_jobs=-1)(
                    delayed(compute_gradient)(iso, p_at_T[i], w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon) 
                    for i, iso in enumerate (self.isoList)
                    )     
                
                # Sum all individual gradients
                temp_grad = {
                    'w1x':0, 
                    'w1y':0,
                    'gz':0,
                    }
                for grad in grad_dict_list: 
                    if optim_dict['w1x']:
                        temp_grad['w1x']    += grad['w1x']*updated_cF['w1x'].scale
                    if optim_dict['w1y']:
                        temp_grad['w1y']    += grad['w1y']*updated_cF['w1y'].scale
                    if optim_dict['gz']:
                        temp_grad['gz']     += grad['gz']*updated_cF['gz'].scale

            case 'min_energy':
                # Compute temporal controls

                rf          = w1x_temp + 1j*w1y_temp
                cost        = self.__dt * np.sum(np.abs(rf/2/np.pi)**2)
                #cost        = self.__dt * np.trapz(np.abs(rf/2/np.pi)**2)
                temp_grad = {'w1x':[], 'w1y':[], 'gz':[]}
                
                if control_fields['w1x'].optimize:
                    wx=np.copy(w1x_temp)    
                #    wx[0]=wx[0]/2
                #    wx[-1]=wx[-1]/2
                    temp_grad['w1x'] = 2 * self.__dt * wx / (4 * np.pi**2)*updated_cF['w1x'].scale
                if control_fields['w1y'].optimize:            
                    wy=np.copy(w1y_temp)    
                #    wy[0]=wy[0]/2
                #    wy[-1]=wy[-1]/2
                    temp_grad['w1y'] = 2 * self.__dt * wy / (4 * np.pi**2)*updated_cF['w1y'].scale
                if control_fields['gz'].optimize:
                    temp_grad['gz'] = -0 * gz_temp
            
            case 'feasibility':
                cost = 1
                grad = 0*x
            case 'min_max_amplitude':
                cost = x[-1]
                grad = np.array([0]*(len(x)-1)+[1])

        # Compute param gradient if cost is not feasibility
        if (self.__costParameters.cost_function != 'feasibility') and (self.__costParameters.cost_function != 'min_max_amplitude'):
                    
            # Project gradient in parameter space
            param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
            
            # Reshape to match X
            grad = np.zeros(x.shape)
            if control_fields['w1x'].optimize:
                grad[self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
            if control_fields['w1y'].optimize:            
                grad[self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
            if control_fields['gz'].optimize:
                grad[self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']).flatten()
            
        if self.__costParameters.cost_function=='min_max_amplitude':
            print("max amplitude = %0.5f (uT)"%np.sqrt(cost))
        else:
            print("Current cost = %0.5f"%cost)
        # print("Current grad = ", grad)
        
        # # Save the current control fields
        # current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # np.savez(self.__sol_folder + f'/b1_it_{current_datetime}.npz', w1x=w1x_temp, w1y=w1y_temp, gz=gz_temp)
        return cost, grad
    
    def compute_final_adjoint_state(self, iso):
        p_at_T = np.zeros(4)
        match self.__costParameters.cost_function:
            case 'distance_to_target':
                p_at_T[1:] = 2/len(self.isoList) * iso['weight'] * (iso['M_at_T'] - iso['target'] ) #__isoList
            case 'feasibility':
                p_at_T = 0*p_at_T 
        return p_at_T
    
    def get_gradient_in_parameter_space(self, control_fields, temp_grad):
        param_grad = {
            'w1x': [], 
            'w1y': [], 
            'gz': [], 
        }
        if control_fields['w1x'].optimize:
            param_grad['w1x'] = control_fields['w1x'].compute_paramGrad(temp_grad['w1x'])
        if control_fields['w1y'].optimize:
            param_grad['w1y'] = control_fields['w1y'].compute_paramGrad(temp_grad['w1y'])
        if control_fields['gz'].optimize:
            param_grad['gz'] = control_fields['gz'].compute_paramGrad(temp_grad['gz'])
        
        return param_grad
   
   ############### CONSTRAINTS ################ 
    def constraints_with_propagation(self,x,control_fields):
        updated_cF      = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        updated_isoList = propagation.propagate_par(updated_cF, self.isoList, self.__global)
        Constraints_used =self.__constraints.keys()
        w1x_temp    = updated_cF['w1x'].compute_tempShape()*updated_cF['w1x'].scale
        w1y_temp    = updated_cF['w1y'].compute_tempShape()*updated_cF['w1y'].scale
        gz_temp      = updated_cF['gz'].compute_tempShape()*updated_cF['gz'].scale
        w0_mat      = self.__w0_mat
        w1x_mat     = self.__w1x_mat
        w1y_mat     = self.__w1y_mat
        optim_dict = {
            'w1x':updated_cF['w1x'].optimize,
            'w1y':updated_cF['w1y'].optimize,
            'gz':updated_cF['gz'].optimize}
        constraints=[]
        gradients= []
        if 'Mxy_inSlice_lb' in Constraints_used:
            if 'isoList_comp' in self.__constraints['Mxy_inSlice_lb'].keys():
                df_aux=Analysis.plot_final_magn(isoList=self.__constraints['Mxy_inSlice_lb']['isoList_comp'],plot=False)
                self.__constraints['Mxy_inSlice_lb']['bound'] =df_aux['Transverse magn.'].to_numpy()
            elif isinstance(self.__constraints['Mxy_inSlice_lb']['bound'],numbers.Number):
                self.__constraints['Mxy_inSlice_lb']['bound'] = self.__constraints['Mxy_inSlice_lb']['bound']*np.ones(len(self.isoList))

            bnds=self.__constraints['Mxy_inSlice_lb']['bound']
            #Constraint
            for i,iso in enumerate(updated_isoList):
                if iso['slice_status']=='in':
                    constraints=constraints+[iso['M_at_T'][0]**2+iso['M_at_T'][1]**2-bnds[i]**2]
            
            #Gradient 
            # Get final adjoint state for all isochromats
            p_at_T = []
            ncons=0
            for iso in updated_isoList:
                p_at_T_aux=np.zeros(4)
                if iso['slice_status']=='in':
                    p_at_T_aux[1] = 2 * iso['M_at_T'][0]
                    p_at_T_aux[2] = 2 * iso['M_at_T'][1]
                    ncons +=1          
                p_at_T.append( p_at_T_aux )
                    
            # Compute gradient using parallel computation
            grad_dict_list = []
            grad_dict_list = Parallel(n_jobs=-1)(
                delayed(compute_gradient)(iso, p_at_T[i], w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon) 
                for i, iso in enumerate (updated_isoList) if iso['slice_status']=='in'
                )     
                
            grads = np.zeros((ncons,len(x)))
            i=0
            for grad in grad_dict_list: 
            # print(grad['iso_pos'])
                temp_grad = {
                    'w1x':0, 
                    'w1y':0,
                    'gz':0,
                    }
                if optim_dict['w1x']:
                    temp_grad['w1x']    = grad['w1x']*updated_cF['w1x'].scale
                if optim_dict['w1y']:
                    temp_grad['w1y']    = grad['w1y']*updated_cF['w1y'].scale
                if optim_dict['gz']:
                    temp_grad['gz']     = grad['gz']*updated_cF['gz'].scale
                    
                # Project gradient in parameter space
                param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
            
            # Reshape to match X

                if control_fields['w1x'].optimize:
                    grads[i,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
                if control_fields['w1y'].optimize:            
                    grads[i,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()#param_grad['w1y'][:,0]
                if control_fields['gz'].optimize:
                    grads[i,self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']).flatten()#param_grad['gz'][:,0]
                
                i+=1
            gradients=gradients+grads.tolist()

        if 'Mxy_lb' in Constraints_used:
            for iso in updated_isoList:
                # Constraint
                constraints=constraints+[( iso['M_at_T'][0]**2 + iso['M_at_T'][1]**2  - iso['Mxy_lb']**2)]
                
                #Gradient
                p_at_T = np.zeros(4)
                p_at_T[1:] = 2 * iso['M_at_T']
                p_at_T[3] = 0
                # Compute temporal gradient from final adjoint state
                temp_grad = compute_gradient(iso, p_at_T, w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon)        
                        
                # Project gradient in parameter space
                param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
                
                # Reshape to match X
                grad = np.zeros(x.shape)
                if control_fields['w1x'].optimize:
                    grad[self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']*updated_cF['w1x'].scale).flatten()
                if control_fields['w1y'].optimize:            
                    grad[self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']*updated_cF['w1y'].scale).flatten()
                if control_fields['gz'].optimize:
                    grad[self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']*updated_cF['gz'].scale).flatten()

                gradients=gradients+grad.tolist()         

        if 'Mxy_outSlice_ub' in Constraints_used:
            if 'isoList_comp' in self.__constraints['Mxy_outSlice_ub'].keys():
                df_aux=Analysis.plot_final_magn(isoList=self.__constraints['Mxy_inSlice_lb']['isoList_comp'],plot=False)
                self.__constraints['Mxy_inSlice_lb']['bound'] =df_aux['Transverse magn.'].to_numpy()
            elif isinstance(self.__constraints['Mxy_outSlice_ub']['bound'],numbers.Number):
                bound_value=self.__constraints['Mxy_outSlice_ub']['bound']
                self.__constraints['Mxy_outSlice_ub']['bound'] = bound_value*np.ones(self.Nout)

            #Constraints
            for iso in updated_isoList:
                if iso['slice_status']=='out':
                    constraints=constraints+[100*(iso['Mxy_outSlice_ub']**2-iso['M_at_T'][0]**2-iso['M_at_T'][1]**2)] ###Atention iso['M_at_T'] only save the M vector
            
            #Gradient
            p_at_T = []
            ncons=0
            for iso in updated_isoList:
                p_at_T_aux=np.zeros(4)
                if iso['slice_status']=='out':
                    p_at_T_aux[1] = -200* iso['M_at_T'][0]
                    p_at_T_aux[2] = -200* iso['M_at_T'][1]
                    ncons +=1          
                p_at_T.append( p_at_T_aux )      
            # Compute gradient using parallel computation
            grad_dict_list = []
            grad_dict_list = Parallel(n_jobs=-1)(
                delayed(compute_gradient)(iso, p_at_T[i], w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon) 
                for i, iso in enumerate (updated_isoList) if iso['slice_status']=='out'
                )     
                
            grads = np.zeros((ncons,len(x)))
            i=0
            for grad in grad_dict_list: 
            # print(grad['iso_pos'])
                temp_grad = {
                    'w1x':0, 
                    'w1y':0,
                    'gz':0,
                    }
                if optim_dict['w1x']:
                    temp_grad['w1x']    = grad['w1x']*updated_cF['w1x'].scale
                if optim_dict['w1y']:
                    temp_grad['w1y']    = grad['w1y']*updated_cF['w1y'].scale
                if optim_dict['gz']:
                    temp_grad['gz']     = grad['gz']*updated_cF['gz'].scale
                    
                # Project gradient in parameter space
                param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
            
                # Reshape to match X
                if control_fields['w1x'].optimize:
                    grads[i,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
                if control_fields['w1y'].optimize:            
                    grads[i,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
                if control_fields['gz'].optimize:
                    grads[i,self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']).flatten()
                
                i+=1    
            gradients=gradients+grads.tolist()

        if 'signal_in' in Constraints_used:
            if 'isoList_comp' in self.__constraints['signal_in'].keys():
                isoList_comp=propagation.compute_refoc(self.__constraints['signal_in']['isoList_comp'],gz_temp[0],self.__refoc_f*self.__T_s ,self.__gamma)
                vec_in=[isoList_comp[i]['M_refoc'][0:2] for i in range(len(isoList_comp)) if isoList_comp[i]['slice_status']=='in']
                sum_vec=sum(vec_in)
                self.__constraints['signal_in']['bound']=(sum_vec[0]**2+sum_vec[1]**2)/len(vec_in)**2
            updated_isoList = propagation.compute_refoc(updated_isoList,gz_temp[0],self.__refoc_f*self.__T_s  ,self.__gamma)
            bound=self.__constraints['signal_in']['bound']
            alpha=self.__constraints['signal_in']['alpha']
            vec_in=[updated_isoList[i]['M_refoc'][0:2] for i in range(len(updated_isoList)) if updated_isoList[i]['slice_status']=='in']
            sum_vec=sum(vec_in)
            # Constraint
            constraints=constraints+[(sum_vec[0]**2+sum_vec[1]**2)/len(vec_in)**2-alpha**2*bound]
            
            # Gradient
            p_at_T = []

            vec_in=[updated_isoList[i]['M_refoc'][0:2] for i in range(len(updated_isoList)) if updated_isoList[i]['slice_status']=='in']
            sum_vec=sum(vec_in)
            for iso in updated_isoList:
                p_at_T_aux=np.zeros(4)
                ang_r=-(self.__gamma*gz_temp[0]*1e-3*self.__refoc_f*self.__T_s*iso['pos_mm'])
                if iso['slice_status']=='in':
                    p_at_T_aux[1] = 2/len(vec_in)**2*sum_vec[0]*np.cos(ang_r)-2/len(vec_in)**2*sum_vec[1]*np.sin(ang_r)
                    p_at_T_aux[2] = 2/len(vec_in)**2*sum_vec[0]*np.sin(ang_r)+2/len(vec_in)**2*sum_vec[1]*np.cos(ang_r)      
                p_at_T.append( p_at_T_aux )

            grad_dict_list = []
            grad_dict_list = Parallel(n_jobs=-1)(
                delayed(compute_gradient)(iso, p_at_T[i], w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon) 
                for i, iso in enumerate (updated_isoList) if iso['slice_status']=='in'
                ) 

            temp_grad = {
                'w1x':0, 
                'w1y':0,
                'gz' :0,
                }

            for grad in grad_dict_list: 
                if optim_dict['w1x']:
                    temp_grad['w1x']    = temp_grad['w1x'] + grad['w1x']*updated_cF['w1x'].scale
                if optim_dict['w1y']:
                    temp_grad['w1y']    = temp_grad['w1y'] + grad['w1y']*updated_cF['w1y'].scale
                if optim_dict['gz']:
                    temp_grad['gz']     = temp_grad['gz']  + grad['gz']*updated_cF['gz'].scale
                    
            param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
            grad = np.zeros((1,len(x)))
            # Reshape to match X
            if control_fields['w1x'].optimize:
                grad[0,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
            if control_fields['w1y'].optimize:            
                grad[0,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
            if control_fields['gz'].optimize:
                grad[0,self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']).flatten()   
            
            gradients=gradients+grad.tolist()

        if 'Mz_inSlice_ub' in Constraints_used:
            if 'isoList_comp' in self.__constraints['Mz_inSlice_ub'].keys():
                df_aux=Analysis.plot_final_magn(isoList=self.__constraints['Mxy_inSlice_lb']['isoList_comp'],plot=False)
                self.__constraints['Mxy_inSlice_lb']['bound'] =df_aux['Transverse magn.'].to_numpy()
            elif isinstance(self.__constraints['Mz_inSlice_ub']['bound'],numbers.Number):
                bound_value=self.__constraints['Mz_inSlice_ub']['bound']
                self.__constraints['Mz_inSlice_ub']['bound'] = bound_value*np.ones(self.Nin)

            #Constraint
            for iso in updated_isoList:
                if iso['slice_status']=='in':
                    constraints = constraints + [ 1*( iso['Mz_inSlice_ub'] - iso['M_at_T'][2] ) ] 
            #Gradient
            p_at_T = []
            ncons = 0
            for iso in updated_isoList:
                p_at_T_aux=np.zeros(4)
                if iso['slice_status']=='in':
                    p_at_T_aux[3] = -1
                    ncons +=1          
                p_at_T.append( p_at_T_aux )      
            # Compute gradient using parallel computation
            grad_dict_list = []
            grad_dict_list = Parallel(n_jobs=-1)(
                delayed(compute_gradient)(iso, p_at_T[i], w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, self.__dt, self.__gamma, optim_dict , self.__epsilon) 
                for i, iso in enumerate (updated_isoList) if iso['slice_status']=='in'
                )     
                
            grads = np.zeros((ncons,len(x)))
            i=0
            for grad in grad_dict_list: 
            # print(grad['iso_pos'])
                temp_grad = {
                    'w1x':0, 
                    'w1y':0,
                    'gz':0,
                    }
                if optim_dict['w1x']:
                    temp_grad['w1x']    = grad['w1x']*updated_cF['w1x'].scale
                if optim_dict['w1y']:
                    temp_grad['w1y']    = grad['w1y']*updated_cF['w1y'].scale
                if optim_dict['gz']:
                    temp_grad['gz']     = grad['gz']*updated_cF['gz'].scale
                    
                # Project gradient in parameter space
                param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
            
                # Reshape to match X
                if control_fields['w1x'].optimize:
                    grads[i,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
                if control_fields['w1y'].optimize:            
                    grads[i,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
                if control_fields['gz'].optimize:
                    grads[i,self.__idx_X['gz'][0]  : self.__idx_X['gz'][1]]    = np.array(param_grad['gz']).flatten()
                
                i+=1
            gradients=gradients+grads.tolist()

        return np.array(constraints), np.array(gradients)

    def constraint_pulse_energy(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        
        rf              = w1x_temp + 1j*w1y_temp
        E_Hz           = self.__dt * np.sum(np.abs(rf/2/np.pi)**2)
        ineq_constraint = self.__constraints['max_energy']['bound'] - E_Hz
        
        return ineq_constraint
    
    def constraint_gradient_pulse_energy(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        gz_temp     = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape()
        # could be shared with constraint_pulse_energy
        
        temp_grad = {'w1x':np.zeros(w1x_temp.size), 'w1y':np.zeros(w1y_temp.size),'gz':np.zeros(gz_temp.size)} #gz need it in get_gradient_in_parameter_space when it is optimized
        if control_fields['w1x'].optimize:
            temp_grad['w1x'] = -2 * self.__dt * w1x_temp / (4 * np.pi**2)*updated_cF['w1x'].scale
        if control_fields['w1y'].optimize:
            temp_grad['w1y'] = -2 * self.__dt * w1y_temp / (4 * np.pi**2)*updated_cF['w1y'].scale
       
                    
        # Project gradient in parameter space
        param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
        
        # Reshape to match X
        grad = np.zeros(x.shape)
        if control_fields['w1x'].optimize:
            grad[self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
        if control_fields['w1y'].optimize:            
            grad[self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
        
        return grad

    # set true Jac and write that in only one function    
  
    def constraint_max_peak_amplitude_uT(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        cons=[]
        for idx in range(len(w1x_temp)):
            peak_square_uT_vec = ( w1x_temp[idx]**2 + w1y_temp[idx]**2 ) / (self.__gamma * 1e-6)**2
            cons += [self.__constraints['max_peak_amplitude_uT']['bound']**2 - peak_square_uT_vec]
        
        return np.array(cons)
    
    def constraint_gradient_max_peak_amplitude_uT(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        gz_temp     = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape()

        grads = np.zeros((len(w1x_temp),len(x)))
        for idx in range(len(w1x_temp)):
            temp_grad = {'w1x':np.zeros(w1x_temp.size), 'w1y':np.zeros(w1y_temp.size),'gz':np.zeros(gz_temp.size)}
            if control_fields['w1x'].optimize:
                temp_grad['w1x'][idx] = -2 * w1x_temp[idx] / (self.__gamma * 1e-6)**2*updated_cF['w1x'].scale
            if control_fields['w1y'].optimize:
                temp_grad['w1y'][idx] = -2 * w1y_temp[idx] / (self.__gamma * 1e-6)**2*updated_cF['w1y'].scale
        
            # Project gradient in parameter space
            param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
        
            # Reshape to match X
            if control_fields['w1x'].optimize:
                grads[idx,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
            if control_fields['w1y'].optimize:            
                grads[idx,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
        
        return grads
    
############ OOOOOOX

    def constraint_max_peak_amplitude_g(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        g_temp    = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape().squeeze()
        cons=[]
        for idx in range(len(g_temp)):
            peak_square_uT_vec = g_temp[idx]**2
            cons += [self.__constraints['max_amplitude_g']['bound']**2 - peak_square_uT_vec]
        
        return np.array(cons)
    
    def constraint_gradient_max_peak_amplitude_g(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x, control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        g_temp     = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape()

        grads = np.zeros((len(g_temp),len(x)))
        for idx in range(len(w1x_temp)):
            temp_grad = {'w1x':np.zeros(w1x_temp.size), 'w1y':np.zeros(w1y_temp.size),'gz':np.zeros(g_temp.size)}
            if control_fields['gz'].optimize:
                temp_grad['gz'][idx] = -2 * g_temp[idx] *updated_cF['gz'].scale
        
            # Project gradient in parameter space
            param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
        
            # Reshape to match X
            if control_fields['gz'].optimize:
                grads[idx,self.__idx_X['gz'][0] : self.__idx_X['gz'][1]]   = np.array(param_grad['gz']).flatten()        
        return grads

############# OOOOOX

    def constraint_min_max_peak_amplitude_uT(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x[:-1], control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        cons=[]
        for idx in range(len(w1x_temp)):
            peak_square_uT_vec = ( w1x_temp[idx]**2 + w1y_temp[idx]**2 ) / (self.__gamma * 1e-6)**2
            cons += [x[-1] - peak_square_uT_vec]
        
        return np.array(cons)
        
    def constraint_gradient_min_max_peak_amplitude_uT(self, x, control_fields):
        updated_cF  = self.get_controlFields_from_X(X=x[:-1], control_fields=control_fields)
        w1x_temp    = updated_cF['w1x'].scale*updated_cF['w1x'].compute_tempShape().squeeze()
        w1y_temp    = updated_cF['w1y'].scale*updated_cF['w1y'].compute_tempShape().squeeze()
        gz_temp     = updated_cF['gz'].scale*updated_cF['gz'].compute_tempShape().squeeze()

        grads = np.zeros((len(w1x_temp),len(x)))
        for idx in range(len(w1x_temp)):
            temp_grad = {'w1x':np.zeros(w1x_temp.size), 'w1y':np.zeros(w1y_temp.size),'gz':np.zeros(gz_temp.size)}
            if control_fields['w1x'].optimize:
                temp_grad['w1x'][idx] = -2 * w1x_temp[idx] / (self.__gamma * 1e-6)**2*updated_cF['w1x'].scale
            if control_fields['w1y'].optimize:
                temp_grad['w1y'][idx] = -2 * w1y_temp[idx] / (self.__gamma * 1e-6)**2*updated_cF['w1y'].scale
                    
            # Project gradient in parameter space
            param_grad = self.get_gradient_in_parameter_space(control_fields, temp_grad)
        
            # Reshape to match X
            if control_fields['w1x'].optimize:
                grads[idx,self.__idx_X['w1x'][0] : self.__idx_X['w1x'][1]]   = np.array(param_grad['w1x']).flatten()
            if control_fields['w1y'].optimize:            
                grads[idx,self.__idx_X['w1y'][0] : self.__idx_X['w1y'][1]]   = np.array(param_grad['w1y']).flatten()
            grads[idx,-1] =1
        
        return grads

        
    ########## Exports #############

    def export_tempcontrol_uT(self,control_fields, name_folder):
        wx = control_fields['w1x'].compute_tempShape()
        wy = control_fields['w1y'].compute_tempShape()
        gz= control_fields['gz'].compute_tempShape()
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        df= DataFrame()
        df['wx']=wx
        df['wy']=wy
        df['gz']=gz
        df.to_csv(name_folder+'/controls.csv')
        
    def export_final_transverseMagn(self,name_folder):
        trMagn  = []
        mz      = []
        pos_mm  = []
        cs      = []
        b1p     = []
        for i in self.isoList: #__isoList
            trMagn.append(np.abs(i['M_at_T'][0] + 1j*i['M_at_T'][1]))
            mz.append(i['M_at_T'][2])
            pos_mm.append(i['pos_mm'])
            cs.append(i['CS']/2/np.pi)
            b1p.append(i['B1p'])

        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
        df = DataFrame()
        df['Transverse magn.'] = trMagn
        df['Longitudinal magn.'] = mz
        df['Position (mm)'] = pos_mm
        df['B1 factor'] = b1p
        df['Resonance offset (Hz)'] = cs
        df.to_csv(name_folder+'/transverseMag.csv')

################################      

def distance_to_target(x, target, weight=1.0):
    return weight * (x - target)@(x - target)


def compute_gradient(iso, p_at_T, w1x_temp, w1y_temp, gz_temp, w1x_mat, w1y_mat, w0_mat, dt, gamma, optim_dict , epsilon):
    # Construct the T_mat matrix based on iso parameters
    T_mat = np.array([
        [0, 0, 0, 0],
        [0, -1/iso['T2'], 0, 0],
        [0, 0, -1/iso['T2'], 0],
        [iso['PD']/iso['T1'], 0, 0, -1/iso['T1']]
    ])
    # Calculate w0_rads
    if not iso['motion_props']:
        motion_mm = np.zeros(gz_temp.size)
    else:
        motion_mm = iso['motion_props']['motion']
        
    w0_rads = iso['CS'] + (iso['pos_mm'] + motion_mm) * 1e-3 * gamma * gz_temp
    
    # Construct propagation matrix for gradient calculation if required
    # Forward propagation
    mat_list = [propagation.taylor_matrix_exponential(  (iso['B1p']*w1x_mat*w1x_temp[i] + iso['B1p']*w1y_mat*w1y_temp[i] + w0_mat*w0_rads[i] + T_mat) *dt ) for i in np.arange(len(w1x_temp))]
    
    # Gradient matrix for w1x
    if optim_dict['w1x']:
        mat_list_w1x = [propagation.taylor_matrix_exponential_cplx(  (iso['B1p']*w1x_mat*(w1x_temp[i]+1j*epsilon) + iso['B1p']*w1y_mat*w1y_temp[i] + w0_mat*w0_rads[i] + T_mat) *dt ) for i in np.arange(len(w1x_temp))]
    if optim_dict['w1y']:
        mat_list_w1y = [propagation.taylor_matrix_exponential_cplx(  (iso['B1p']*w1x_mat*w1x_temp[i] + iso['B1p']*w1y_mat*(w1y_temp[i]+1j*epsilon) + w0_mat*w0_rads[i] + T_mat) *dt ) for i in np.arange(len(w1x_temp))]
    if optim_dict['gz']:
        mat_list_gz = [propagation.taylor_matrix_exponential_cplx(  (iso['B1p']*w1x_mat*w1x_temp[i] + iso['B1p']*w1y_mat*w1y_temp[i] + w0_mat*(iso['CS'] + (iso['pos_mm'] + motion_mm[i])*1e-3*gamma*(gz_temp[i]+1j*epsilon)) + T_mat) *dt ) for i in np.arange(len(w1x_temp))]

    # Compute adjoint state for all times 
    dot = np.dot
    p = [p_at_T]
    for i in np.flip(np.arange(len(w1x_temp))):
        p.insert(0, dot(p[0] , mat_list[i]) )

    # Compute gradient
    grad_dict = {'iso_pos':iso['pos_mm'],
        'w1x':np.zeros(w1x_temp.size), 
        'w1y':np.zeros(w1x_temp.size), 
        'gz':np.zeros(w1x_temp.size)
        }
    for i in np.flip(np.arange(len(w1x_temp))):
        if optim_dict['w1x']:
            grad_dict['w1x'][i] = 1/epsilon*np.imag(dot(p[i+1], dot(mat_list_w1x[i], iso['M'][i])))
        if optim_dict['w1y']:
            grad_dict['w1y'][i] = 1/epsilon*np.imag(dot(p[i+1], dot(mat_list_w1y[i], iso['M'][i])))
        if optim_dict['gz']:
            grad_dict['gz'][i] = 1/epsilon*np.imag(dot(p[i+1], dot(mat_list_gz[i], iso['M'][i])))
    
    return grad_dict