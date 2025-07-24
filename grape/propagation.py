import numpy as np
import plotly.express as px
from scipy.linalg import expm
import copy
from numba import jit
from joblib import Parallel, delayed
import pandas as pd

def propagate_par(control_fields,isoList,global_parameters):
    w1x_mat = np.array( [ [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 1.],
                            [0., 0., -1., 0.] ])
    w1y_mat = np.array( [ [0., 0., 0., 0.],
                            [0., 0., 0., -1.],
                            [0., 0., 0., 0.],
                            [0., 1., 0., 0.] ])
    w0_mat = np.array( [ [0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., -1., 0., 0.],
                            [0., 0, 0., 0.] ])
    w1x_temp    = control_fields['w1x'].scale*control_fields['w1x'].compute_tempShape()
    w1y_temp    = control_fields['w1y'].scale*control_fields['w1y'].compute_tempShape()
    g_temp      = control_fields['gz'].scale*control_fields['gz'].compute_tempShape()
    dt  = global_parameters.t_s[1] - global_parameters.t_s[0]
    gammas                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}
    gamma=gammas[global_parameters.nucleus]
    output_isoList = []
    output_isoList = Parallel(n_jobs=-1)(
        delayed(propagate_iso)(iso, w1x_temp, w1y_temp, w1x_mat, w1y_mat, w0_mat, dt, gamma, g_temp, return_M=True) 
        for iso in isoList
    )

    return output_isoList

def propagate_iso(iso, w1x_temp, w1y_temp, w1x_mat, w1y_mat, w0_mat, dt, gamma, g_temp, return_M=False):
    # Construct the T_mat matrix based on iso parameters
    o_iso = copy.deepcopy(iso)
    T_mat = np.array([
        [0, 0, 0, 0],
        [0, -1/iso['T2'], 0, 0],
        [0, 0, -1/iso['T2'], 0],
        [iso['PD']/iso['T1'], 0, 0, -1/iso['T1']]
        ])
    
    # Calculate w0_rads
    if not iso['motion_props']:
        motion_mm = 0
    else:
        motion_mm = iso['motion_props']['motion']
    
    w0_rads = iso['CS'] + (iso['pos_mm'] + motion_mm) * 1e-3 * gamma * g_temp
    
    # Calculate mat_list using taylor_matrix_exponential
    b1p = iso['B1p']

    #mat_list = [
    #    taylor_matrix_exponential((b1p * w1x_mat * w1x_temp[t] + b1p * w1y_mat * w1y_temp[t] + w0_mat * w0_rads[t] + T_mat) * dt)
        # expm(  (b1p *w1x_mat*w1x_temp[t] + b1p *w1y_mat*w1y_temp[t] + w0_mat*w0_rads[t] + T_mat) *dt )
    #    for t in np.arange(len(w1x_temp))
    #]
    
    # Initialize M and perform the dot product
    # M = [np.array([1, 0, 0, iso['M0']])]
    M = [np.hstack(([1], iso['M0']))]
    for t in np.arange(len(w1x_temp)):
        M.append(np.dot(taylor_matrix_exponential((b1p * w1x_mat * w1x_temp[t] + b1p * w1y_mat * w1y_temp[t] + w0_mat * w0_rads[t] + T_mat) * dt), M[t]))
    
    # Store the result in iso
    o_iso['M_at_T'] = M[-1][1:]  # Get the values after the first element
    if return_M:
        o_iso['M'] = M
    return o_iso

def propagate_par_approx_rot(control_fields,isoList,global_parameters):
    # Should vectorize all trajectories to avoid loop on isoList
    w1x_temp    = control_fields['w1x'].compute_tempShape()
    w1y_temp    = control_fields['w1y'].compute_tempShape()
    g_temp      = control_fields['gz'].compute_tempShape()
    dt  = global_parameters.t_s[1] - global_parameters.t_s[0]
    gammas                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}
    gamma=gammas[global_parameters.nucleus]
    # output_isoList = []
    # output_isoList = Parallel(n_jobs=-1)(
    #     delayed(propagate_iso_approx_rot)(iso['motion_props']['motion'], iso['CS'], iso['pos_mm'], 
    #                                       iso['B1p'], iso['M0'], iso['T1'], iso['T2'], w1x_temp, w1y_temp, dt, gamma, g_temp, return_M=True) 
    #     for iso in isoList
    # )

    np.meshgrid()

    M_list = []
    M_list = Parallel(n_jobs=-1)(
        delayed(propagate_iso_approx_rot)(iso['motion_props']['motion'], iso['CS'], iso['pos_mm'], 
                                          iso['B1p'], iso['M0'], iso['T1'], iso['T2'], w1x_temp, w1y_temp, dt, gamma, g_temp, return_M=True) 
        for iso in isoList
    )

    for idx, iso in enumerate(isoList):
        # Update the iso object with the propagated M
        iso['M_at_T'] = M_list[idx][-1]  # Get the values after the first element
        iso['M'] = M_list[idx] 

    return isoList

@jit(nopython=True) 
def propagate_iso_approx_rot(motion, cs, pos_mm, b1p, M0, T1, T2,  w1x_temp, w1y_temp, dt, gamma, g_temp, return_M=False):
    # o_iso = copy.deepcopy(iso)
    
    # # Calculate w0_rads
    # if not iso['motion_props']:
    #     motion_mm = 0
    # else:
    #     motion_mm = iso['motion_props']['motion']
    
    # w0_rads = iso['CS'] + (iso['pos_mm'] + motion_mm) * 1e-3 * gamma * g_temp
    
    # # Calculate mat_list using taylor_matrix_exponential
    # b1p = iso['B1p']

    # # Initialize M and perform the dot product
    # M = [iso['M0']]

    w0_rads = cs + (pos_mm + motion) * 1e-3 * gamma * g_temp

    # Initialize M and perform the dot product
    M = [M0.astype(np.float64)]

    for t in np.arange(len(w1x_temp)):
        Uz = w0_rads[t]
        Ux = b1p*w1x_temp[t]
        Uy = b1p*w1y_temp[t]

        phy = np.arctan2(Ux, Uy) # the difference
        the = np.arctan2(np.sqrt(Ux**2 + Uy**2),Uz)
        chi = np.sqrt(Ux**2 + Uy**2 + Uz**2) * dt

        cos_phy = np.cos(phy)
        sin_phy = np.sin(phy)
        cos_the = np.cos(the)
        sin_the = np.sin(the)
        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)

        # Rz + phy
        M_prev = np.asarray(M[-1], dtype=np.float64)
        M_x = cos_phy * M_prev[0] + sin_phy * M_prev[1]
        M_y = -sin_phy * M_prev[0] + cos_phy * M_prev[1]
        M_z = M_prev[2]

        # Ry + the
        M_prev = np.array([M_x, M_y, M_z])
        M_x = cos_the * M_prev[0] - sin_the * M_prev[2]
        M_z = sin_the * M_prev[0] + cos_the * M_prev[2]

        # Rz - chi
        M_prev = np.array([M_x, M_y, M_z])
        M_x = cos_chi * M_prev[0] - sin_chi * M_prev[1]
        M_y = sin_chi * M_prev[0] + cos_chi * M_prev[1]

        # Ry - the
        M_prev = np.array([M_x, M_y, M_z])
        M_x = cos_the * M_prev[0] + sin_the * M_prev[2]
        M_z = -sin_the * M_prev[0] + cos_the * M_prev[2]

        # Rz - phy
        M_prev = np.array([M_x, M_y, M_z])
        M_x = cos_phy * M_prev[0] - sin_phy * M_prev[1]
        M_y = sin_phy * M_prev[0] + cos_phy * M_prev[1]

        # Relaxation
        # if iso['T2'] > 0:
        #     M_x *= np.exp(-dt / iso['T2'])
        #     M_y *= np.exp(-dt / iso['T2'])
        # if iso['T1'] > 0:
        #     M_z = (M_z - iso['M0'][2]) * np.exp(-dt / iso['T1']) + iso['M0'][2]

        if T2 > 0:
            M_x *= np.exp(-dt / T2)
            M_y *= np.exp(-dt / T2)
        if T1 > 0:
            M_z = (M_z - M0[2]) * np.exp(-dt / T1) + M0[2]

        # Update M
        M.append(np.array([M_x, M_y, M_z]))
    
    # Store the result in iso
    # o_iso['M_at_T'] = np.array([M_x, M_y, M_z])  # Get the values after the first element
    # if return_M:
    #     o_iso['M'] = M
    # return o_iso
    return M

    '''MATLAB CODE FROM '''
    '''for t = 2:length(self.rf_pulse.time)

        dt = self.rf_pulse.time(t) - self.rf_pulse.time(t-1);

        Uz = (Zgrid * self.rf_pulse.GZ(t-1) + Bgrid*self.B0) * self.rf_pulse.gamma;
        Ux = self.rf_pulse.gamma * B1real(t-1);
        Uy = self.rf_pulse.gamma * B1imag(t-1);

        phy = atan2(               Uy , Ux);
        the = atan2(sqrt(Ux.^2+Uy.^2) , Uz);
        chi = sqrt(Ux.^2+Uy.^2+Uz.^2) * dt;

        cos_phy = cos(phy);
        sin_phy = sin(phy);
        cos_the = cos(the);
        sin_the = sin(the);
        cos_chi = cos(chi);
        sin_chi = sin(chi);

        % Rz + phy
        m(:,1,t) =  cos_phy .* m(:,1,t-1) + sin_phy .* m(:,2,t-1);
        m(:,2,t) = -sin_phy .* m(:,1,t-1) + cos_phy .* m(:,2,t-1);
        m(:,3,t) = m(:,3,t-1);

        % Ry + the
        Mprev = m(:,:,t);
        m(:,1,t) =  cos_the .* Mprev(:,1) - sin_the .* Mprev(:,3);
        m(:,3,t) =  sin_the .* Mprev(:,1) + cos_the .* Mprev(:,3);

        % Rz - chi
        Mprev = m(:,:,t);
        m(:,1,t) =  cos_chi .* Mprev(:,1) - sin_chi .* Mprev(:,2);
        m(:,2,t) =  sin_chi .* Mprev(:,1) + cos_chi .* Mprev(:,2);

        % Ry - the
        Mprev = m(:,:,t);
        m(:,1,t) =  cos_the .* Mprev(:,1) + sin_the .* Mprev(:,3);
        m(:,3,t) = -sin_the .* Mprev(:,1) + cos_the .* Mprev(:,3);

        % Rz - phy
        Mprev = m(:,:,t);
        m(:,1,t) =  cos_phy .* Mprev(:,1) - sin_phy .* Mprev(:,2);
        m(:,2,t) =  sin_phy .* Mprev(:,1) + cos_phy .* Mprev(:,2);

        % Relaxation
        % !!! Separation of Rotation THEN Relaxation induce an error linear with 'dt' !!!
        if use_T2_relaxiation
            m(:,1,t) = m(:,1,t)              .* exp( -dt / self.T2 );
            m(:,2,t) = m(:,2,t)              .* exp( -dt / self.T2 );
        end
        if use_T1_relaxiation
            m(:,3,t) =(m(:,3,t) - self.M0.z) .* exp( -dt / self.T1 ) + self.M0.z;
        end

    end % time'''

def generate_isoList(population_list, geo_parameters,global_parameters):
    isoList = []
    pos_mm=compute_positions(geo_parameters)
    for h, pos in enumerate(pos_mm):
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
                                    if np.abs(pos) <= (geo_parameters.sliceThickness_mm)/2:
                                        iso['slice_status']    = 'in'
                                    elif np.abs(pos) < (geo_parameters.sliceThickness_mm)/2 + geo_parameters.transitionWidth_mm:
                                        iso['slice_status']    = 'trans'
                                    else:
                                        iso['slice_status']    = 'out'

                                    m = np.zeros(global_parameters.N)
                                    iso['motion_props']['motion'] = m

                                    isoList.append(iso)
    return isoList

def compute_positions(geo_parameters):
    # Set-up postion vector
    tW      = geo_parameters.transitionWidth_mm
    sT      = geo_parameters.sliceThickness_mm
    # dz      = optim_parameters['geometry'].stepSlice_mm
    maxZ    = geo_parameters.max_stopBand_mm
    nb_TB   = geo_parameters.nb_transBand
    nb_SB   = geo_parameters.nb_stopBand
    N_in    = geo_parameters.nb_in_slice
    if geo_parameters.logStep:
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
    if geo_parameters.halfPos:
        p = p[p >= 0]
    
    return p

@jit(nopython=True) 
def taylor_matrix_exponential(A, terms=20):
    result = np.eye(A.shape[0])  # Start with the identity matrix
    A_power = np.eye(A.shape[0])  # A^0 = I
    factorial = 1

    # Add terms up to the desired number of terms in the series
    for n in range(1, terms):
        A_power = A_power @ A  # Compute A^n
        factorial *= n  # Compute n!
        result += A_power / factorial  # Add the next term in the series

    return result
@jit(nopython=True)        
def taylor_matrix_exponential_cplx(A, terms=20):
    A = A.astype(np.complex128)
    result = np.eye(A.shape[0],dtype=np.complex128)  # Start with the identity matrix
    A_power = np.eye(A.shape[0],dtype=np.complex128)  # A^0 = I
    factorial = 1

    # Add terms up to the desired number of terms in the series
    for n in range(1, terms):
        A_power = A_power @ A  # Compute A^n
        factorial *= n  # Compute n!
        result += A_power / factorial  # Add the next term in the series
    
    return result  

def compute_refoc(isoList,g_refoc,phaseFactor_s,gamma):
    #Mrefoc=np.zeros(len(isoList),3)
    for i in isoList:
        ang_r=-(gamma*g_refoc*1e-3*phaseFactor_s*i['pos_mm'])
        MT=i['M_at_T']
        i['M_refoc']=copy.deepcopy(i['M_at_T'])
        i['M_refoc'][0] = ( np.cos(ang_r)*MT[0] + np.sin(ang_r)*MT[1])
        i['M_refoc'][1] = (-np.sin(ang_r)*MT[0] + np.cos(ang_r)*MT[1])
    return isoList 


