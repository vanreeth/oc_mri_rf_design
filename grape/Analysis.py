import numpy as np
import os
import plotly.express as px
from grape import propagation
import pandas as pd

def disp_rfAnalysis(control_fields,global_parameters,dir,name):
    w1x_temp    = control_fields['w1x'].compute_tempShape()
    w1y_temp    = control_fields['w1y'].compute_tempShape()
    rf          = w1x_temp + 1j*w1y_temp   
    gammas                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}
    gamma=gammas[global_parameters.nucleus]
    tf_s=global_parameters.T_s  
    info={}
    info['max_uT']  = max(np.abs(rf/(gamma*1e-6)))
    info['Power']   = 1/(len(rf)-1)*(np.trapz(np.abs(rf/(2*np.pi))**2))
    info['E_kHz']   = tf_s*info['Power']/1000 ;
    info['S_int']   = np.trapz(w1x_temp/max(np.abs(rf)))/(len(rf)-1)
    info['eff_fa']  = max(np.abs(rf))/(np.pi/(2*0.001)) * tf_s*1e3 * info['S_int'] * 90

    if not os.path.exists(dir):
        os.makedirs(dir)


    f= open(dir+'/'+name+'.txt','w')
    f.write('\n---------- RF ANALYSIS ----------'+'\n'+
    'Average pulse power: '+str(round(info['Power'],3))+'(Hz^2)\n'+
    'Pulse Energy: '+str(round(info['E_kHz'],3))+' (kHz)\n'+
    'Pulse peak amplitude: '+str(round(info['max_uT'],3))+ ' uT\n'+
    'Normalized integral factor: '+str(round(info['S_int'],4))+'\n'+
     'Effective flip angle (degrees): '+str(round(info['eff_fa'],1))+' \n'+
    '---------------------------------\n') 
    f.close()

def plot_Mx_My_circle(isoList, plot=True,refoc={'on':False,'g_refoc':0,'phaseFactor_s':0,'gamma':0}):
    data = []
    # Collect all trajectories
    if refoc['on']:
        isoList=propagation.compute_refoc(isoList,refoc['g_refoc'],refoc['phaseFactor_s'],refoc['gamma'])
        for traj in isoList:
            idx = traj['population_idx']
            mx = np.array(traj['M_refoc'])[0]
            my = np.array(traj['M_refoc'])[1]
            # Append each data point with associated population index
            data.append({'x': mx, 'y': my, 'idx': idx})
    else:
        for traj in isoList:
            idx = traj['population_idx']
            mx = np.array(traj['M_at_T'])[0]
            my = np.array(traj['M_at_T'])[1]
            # Append each data point with associated population index
            data.append({'x': mx, 'y': my, 'idx': idx})
    df = pd.DataFrame(data)
    if plot:
        # Création de la figure
        import plotly.graph_objects as go
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        # Generate a color map based on trajectory_index
        cmap = cm.gist_rainbow
        norm = mcolors.Normalize(vmin=df['idx'].min(), vmax=df['idx'].max())
        # cmap = mcolors.LinearSegmentedColormap.from_list('trajectory_colormap', ['blue', 'green', 'red'])

        fig = go.Figure()
        # Add a circle of radius 1 centered at (0, 0)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",  # Reference the x and y axes
            x0=-1, y0=-1,        # Bottom-left corner of bounding box
            x1=1, y1=1,          # Top-right corner of bounding box
            layer="below",  # Ensure the circle is in the background
            line=dict(color="gray", width=2)  # Circle outline color and width
        )

        # Ajout des trajectoires
        for _, row in df.iterrows():
            x_coords = row['x']
            y_coords = row['y']
            index = row['idx']
            color = mcolors.to_hex(cmap(norm(index)))  # Map index to color
            #fig.add_trace(go.Scatter(
            #    x=x_coords,
            #    y=y_coords,
                #mode='lines',  # Tracer des lignes et des points
            #    showlegend=False,
            #    name=f'Population {row["idx"]}',
            #    line=dict(color=color)
            #))
            # Scatter mark at the last point
            fig.add_trace(go.Scatter(
                x=[x_coords],  # Last x coordinate
                y=[y_coords],  # Last y coordinate
                mode='markers',    # Marker only
                showlegend=False,
                marker=dict(size=7, symbol='circle', color=color)  # Customize marker
            ))
        
        # Add a single legend entry per trajectory_index
        for index in df['idx'].unique():
            color = mcolors.to_hex(cmap(norm(index)))
            fig.add_trace(go.Scatter(
                x=[None],  # Dummy x-coordinate
                y=[None],  # Dummy y-coordinate
                mode='lines+markers',  # Only for legend purposes
                name=f'Population {index}',  # Legend label
                marker=dict(size=5, color=color)  # Color representing the trajectory_index
            ))

        # Mise en forme
        fig.update_layout(
            title="Trajectories by Index",
            xaxis=dict(range=[-1, 1], scaleanchor='y'),  # Link scaling of x-axis to y-axis
            yaxis=dict(range=[-1, 1]),
            xaxis_title="Mx",
            yaxis_title="My",
            legend_title="Population index",
            template="plotly",
            width=600,  # Set width to 800 pixels
            height=600  # Set height to 800 pixels
        )

        # Afficher la figure
        fig.show()
    return df

def plot_final_magn(isoList, x='position', y='M_xy', color='population_idx', plot=True,export=False,folder=''):
    """Generic plot function of the final magnetization state

    Inputs :
        isoList: list of dictionaries containing the properties of each isochrmat (output of generate_isoList)
        x: string to select the x-axis variable - '[position]' or 'cs'
        y: string to select the y-axis variable - '[M_xy]', 'M_z' or 'phase'
        color: string to select the color variable - can be any key of the isochromat dictionary, typically '[population_idx]', 'b1p' or 'CS'
        plot: boolean to display the plot [True]

    Outputs:
      df: DataFrame containing the final magnetization state of each isochromat
    """
    trMagn  = []
    phase   = []
    mz      = []
    pos_mm  = []
    cs      = []
    b1p     = []
    phase   = []
    color_list   = []
    sl_status = []
    for i in isoList: 
        trMagn.append(np.abs(i['M_at_T'][0] + 1j*i['M_at_T'][1]))
        phase.append(np.angle(i['M_at_T'][0] + 1j*i['M_at_T'][1]))
        mz.append(i['M_at_T'][2])
        pos_mm.append(i['pos_mm'])
        cs.append(i['CS']/2/np.pi)
        b1p.append(i['B1p'])
        color_list.append(i[color])
        sl_status.append(i['slice_status'])

    
    df = pd.DataFrame()
    df['Transverse magn.'] = trMagn
    df['Longitudinal magn.'] = mz
    df['Position (mm)'] = pos_mm
    df['B1 factor'] = b1p
    df['Resonance offset (Hz)'] = cs
    df['Phase (rad)'] = np.unwrap(phase)
    df[color] = color_list
    df['slice_status'] = sl_status
    if plot:
        if x == 'position':
            xaxis = 'Position (mm)'
        elif x == 'cs':
            xaxis = 'Resonance offset (Hz)'
        if y == 'M_xy':
            yaxis = 'Transverse magn.'
        elif y == 'M_z':
            yaxis = 'Longitudinal magn.'
        elif y == 'phase':
            yaxis = 'Phase (rad)'
        fig = px.line(df, x = xaxis, y = yaxis, color=color, labels={'x': xaxis, 'y': yaxis}, title='Final magnetization')  
        fig.show()
    
    if export:
        if x == 'position':
            xaxis_name = 'Position_(mm)'
            xaxis=pos_mm
        elif x == 'cs':
            xaxis_name = 'Resonance_offset_(Hz)'
            xaxis= cs
        for co in color_list:
            name1 = 'M_xy_'+str(co)+'.dat'
            name2 = 'M_z_'+str(co)+'.dat'
            name3 = 'phase_'+str(co)+'.dat'
            if len(folder)>0:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                name1=folder+'/'+name1
                name2=folder+'/'+name2
                name3=folder+'/'+name3
            
            np.savetxt(name1,np.transpose([xaxis, trMagn]), header=xaxis_name+' Transverse_magn.',comments='')
            np.savetxt(name2,np.transpose([xaxis, mz]), header=xaxis_name+' Longitudinal_magn',comments='')
            np.savetxt(name3,np.transpose([xaxis, np.unwrap(phase)]), header=xaxis_name+' Phase_(rad)',comments='')
                       
    return df



def plot_targetStates(isoList):
    data = []
    for iso in isoList: 
        complex_target = iso['target'][0] + 1j * iso['target'][1]
        mod = np.abs(complex_target)
        ph  = np.angle(complex_target, deg=True)
        idx = iso['population_idx']
        data.append({'mod': mod, 'ph': ph, 'idx': idx})
    df = pd.DataFrame(data)    
    fig = px.scatter_polar(df, r='mod', theta='ph', title='Target states', color='idx', start_angle=0)
    # Update layout to set direction to counterclockwise
    fig.update_layout(
        polar=dict(
            angularaxis=dict(direction="counterclockwise"),
            radialaxis=dict(showticklabels=False, range=[0, 1])
        )
    )
    fig.show()

def plot_w1_field(global_parameters, control_fields, mode='mp',export=False,folder='.', export_name='B1_field.dat'):
    """
    Plot the RF field

    Inputs:
        global_parameters: object containing the global parameters
        control_fields: dictionary containing the control fields
        mode : 'mp' (default) or 'ri'
        export: boolean to export the plot data to a file [False]
        folder: folder to save the exported file ['.']
        export_name: name of the exported file [B1_field.dat]
    """

    w1x_temp = control_fields['w1x'].compute_tempShape().squeeze()
    w1y_temp = control_fields['w1y'].compute_tempShape().squeeze()
    gammas                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}
    gamma=gammas[global_parameters.nucleus]
    if mode == 'mp':
        px.line(x=global_parameters.t_s*1000, y = np.abs(w1x_temp + 1j * w1y_temp)/gamma*1e6, labels={'x':'Time (ms)', 'y':'Magnitude (µT)'}).show()
        px.line(x=global_parameters.t_s*1000, y = np.unwrap(np.angle(w1x_temp + 1j * w1y_temp)), labels={'x':'Time (ms)', 'y':'Phase (rad)'}).show()
        if export:
            if len(folder)>0:
                export_name=folder+'/'+export_name
                if not os.path.exists(folder):
                    os.makedirs(folder)
            np.savetxt(export_name,np.transpose([global_parameters.t_s*1000, np.abs(w1x_temp + 1j * w1y_temp)/gamma*1e6, np.unwrap(np.angle(w1x_temp + 1j * w1y_temp))]), header='Time_(ms) Magnitude_(uT) Phase_(rad)',comments='')
    elif mode == 'ri':
        px.line(x=global_parameters.t_s*1000, y = w1x_temp/gamma*1e6, labels={'x':'Time (ms)', 'y':'B1x (µT)'}).show()
        px.line(x=global_parameters.t_s*1000, y = w1y_temp/gamma*1e6, labels={'x':'Time (ms)', 'y':'B1y (µT)'}).show()
        if export:
            if len(folder)>0:
                export_name=folder+'/'+export_name
                if not os.path.exists(folder):
                    os.makedirs(folder)
            np.savetxt(export_name,np.transpose([global_parameters.t_s*1000, w1x_temp/gamma*1e6, w1y_temp/gamma*1e6]), header='Time_(ms) B1x_(uT) B1y_(uT)',comments='')
            # np.savetxt(name2,np.transpose([global_parameters.t_s*1000, w1y_temp/gamma*1e6]), header='Time_(ms) B1y_(uT)',comments='')


def plot_trajectories(isoList, plot=True):
    """
    Plot the trajectories of the isochromats in the Mx-My plane

    Inputs:
        isoList: list of dictionaries containing the properties of each isochrmat (output of generate_isoList)
        plot: boolean to display the plot

    Outputs:
        df: DataFrame containing the trajectories of the isochromats in the Mx-My plane
    """
    data = []
    # Collect all trajectories
    for traj in isoList:
        idx = traj['population_idx']
        mx = np.array(traj['M'])[:,1]
        my = np.array(traj['M'])[:,2]
        mz = np.array(traj['M'])[:,3]
        # Append each data point with associated population index
        data.append({'x': mx, 'y': my, 'z':mz, 'idx': idx, 'slice_status': traj['slice_status']})
    df = pd.DataFrame(data)
    if plot:
        # Création de la figure
        import plotly.graph_objects as go
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        # Generate a color map based on trajectory_index
        cmap = cm.gist_rainbow
        norm = mcolors.Normalize(vmin=df['idx'].min(), vmax=df['idx'].max())
        # cmap = mcolors.LinearSegmentedColormap.from_list('trajectory_colormap', ['blue', 'green', 'red'])

        fig = go.Figure()
        # Add a circle of radius 1 centered at (0, 0)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",  # Reference the x and y axes
            x0=-1, y0=-1,        # Bottom-left corner of bounding box
            x1=1, y1=1,          # Top-right corner of bounding box
            layer="below",  # Ensure the circle is in the background
            line=dict(color="gray", width=2)  # Circle outline color and width
        )

        # Ajout des trajectoires
        for _, row in df.iterrows():
            x_coords = row['x']
            y_coords = row['y']
            index = row['idx']
            color = mcolors.to_hex(cmap(norm(index)))  # Map index to color
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',  # Tracer des lignes et des points
                showlegend=False,
                name=f'Population {row["idx"]}',
                line=dict(color=color)
            ))
            # Scatter mark at the last point
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],  # Last x coordinate
                y=[y_coords[-1]],  # Last y coordinate
                mode='markers',    # Marker only
                showlegend=False,
                marker=dict(size=7, symbol='circle', color=color)  # Customize marker
            ))
        
        # Add a single legend entry per trajectory_index
        for index in df['idx'].unique():
            color = mcolors.to_hex(cmap(norm(index)))
            fig.add_trace(go.Scatter(
                x=[None],  # Dummy x-coordinate
                y=[None],  # Dummy y-coordinate
                mode='lines+markers',  # Only for legend purposes
                name=f'Population {index}',  # Legend label
                marker=dict(size=5, color=color)  # Color representing the trajectory_index
            ))

        # Mise en forme
        fig.update_layout(
            title="Trajectories by Index",
            xaxis=dict(range=[-1, 1], scaleanchor='y'),  # Link scaling of x-axis to y-axis
            yaxis=dict(range=[-1, 1]),
            xaxis_title="Mx",
            yaxis_title="My",
            legend_title="Population index",
            template="plotly",
            width=600,  # Set width to 800 pixels
            height=600  # Set height to 800 pixels
        )

        # Afficher la figure
        fig.show()
    return df