import numpy as np
import os
from datetime import datetime

def exportBruker(control_fields,TBP,rephaseFactor,gamma,type,dir,oname):
    """Export the RF shape to a Bruker compatible format.

    Args:
        control_fields (obj): object of the class ControlFields
        TBP (float): Time Bandwidth Product (unitless) > 0
        rephaseFactor (float): in percent
        gamma (float): gyromagnetic ratio
        type (string): excitation (exc), or inversion (inv)
        dir (string): directory where the file will be saved
        oname (string): output name of the RF shape file (without extension)
    """

    #TBP can not be <= 0
    RF={}
    RF['name']=oname
    w1x_temp    = control_fields['w1x'].compute_tempShape()
    w1y_temp    = control_fields['w1y'].compute_tempShape()
    rf          = w1x_temp + 1j*w1y_temp

    RF['amplitude']=np.abs(rf)/max(abs(rf))*100.0
    
    angles=np.angle(rf, deg=True)
    angles[angles<0]=angles[angles<0]+360
    RF['phase']  = angles
    RF['type']    = type

    if type=='exc':
        RF['TOTROT']   = 90
        RF['EXMODE']   = 'Excitation'
        RF['NAME_EXT'] = '.exc'
    elif type=='inv':
            RF['TOTROT']   = 180
            RF['EXMODE']   = 'Inversion'
            RF['NAME_EXT'] = '.inv'        
    #to do the other when needed

    RF['BWFac']           = TBP
    RF['IF']              = np.trapz(w1x_temp/np.max(np.abs(w1x_temp)))/(len(rf)-1) ;
    RF['rephasingfactor'] = rephaseFactor
    RF['B1MaxPower']      = max(abs(rf))/(gamma/1e6) 
    RF['length']          = len(rf)
    RF['Name2save']       = dir+'/'+oname+RF['NAME_EXT'] 

    if not os.path.exists(dir):
            os.makedirs(dir)
    now = datetime.now() 

    f= open(RF['Name2save'],'w')
    f.write('##TITLE= /opt/PV-360.3.4/exp/stan/nmr/lists/wave/'+RF['name']+'\n'+
            '##JCAMP-DX= 5.00 Bruker JCAMP library\n'+
            '##DATA TYPE= Shape Data\n'+
            '##ORIGIN= Creatis UMR CNRS 5220\n'+
            '##OWNER= <Eric>\n'+
            '##DATE= '+now.strftime("%d/%m/%Y")+'\n'+
            '##TIME= '+now.strftime("%H:%M:%S")+'\n'+
            '##MINX= '+"{:10.6e}".format(min(RF['amplitude']))+'\n'+
            '##MAXX= '+"{:10.6e}".format(max(RF['amplitude']))+'\n'+
            '##MINY= '+"{:10.6e}".format(min(RF['phase']))+'\n'+
            '##MAXY= '+"{:10.6e}".format(max(RF['phase']))+'\n'+
            '##$SHAPE_EXMODE= '+RF['EXMODE']+'\n'+
            '##$SHAPE_TOTROT= '+"{:10.6e}".format(RF['TOTROT'])+'\n'+
            '##$SHAPE_BWFAC= '+str(RF['BWFac'])+'\n'+
            '##$SHAPE_INTEGFAC= '+str(round(RF['IF'],5))+'\n'+
            '##$SHAPE_REPHFAC= '+str(round(100*RF['rephasingfactor'],5))+'\n'+
            '##$SHAPE_TYPE= ' +RF['type']+'\n'+
            '##$SHAPE_MODE= 0\n'+
            '##MAX_B1_microT= '+str(round(RF['B1MaxPower'],3))+'\n'+
            '##NPOINTS= '+str(RF['length'])+'\n'+
            '##XYPOINTS= (XY..XY)'+'\n'
                        )
    for i in range(len(RF['amplitude'])):
         f.write("{:10.6e}".format(RF['amplitude'][i])+', '+"{:10.6e}".format(RF['phase'][i])+'\n')
    f.write('##END\n') #finish always with \n at the end

    f.close()

