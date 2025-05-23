import argparse
import logging
from datetime import datetime

from arc import LOG
from arc.Automated_Rating_Curve_Generator import main, get_parameter_name
from arc.Curve2Flood import Curve2Flood_MainFunction

__all__ = ['Arc']

class Arc():
    _mifn: str = ""
    _args: dict = {}
    
    def __init__(self, mifn: str = "", args: dict = {}, quiet: bool = False) -> None:
        self._mifn = mifn
        self._args = args
        self.quiet = quiet
        if quiet:
            self.set_log_level('error')
        
    def run(self):
        global starttime
        global MIF_Name
        starttime = datetime.now()
    
        LOG.info('Inputs to the Program is a Main Input File')
        LOG.info('\nFor Example:')
        LOG.info('  python Automated_Rating_Curve_Generator.py ARC_InputFiles/ARC_Input_File.txt')
        
        ### User-Defined Main Input File ###
        if self._mifn or self._args:
            MIF_Name = self._mifn
            LOG.info('Main Input File Given: ' + MIF_Name)
        else:
            #Read Main Input File
            MIF_Name = 'ARC_InputFiles/ARC_Input_File.txt'
            MIF_Name = r"C:\Projects\2024_FHS_FloodForecasting\ARC_Shields_Nencarta\nencarta_test_wsebathy_clean\yellowstone_wsebathy_clean\ARC_InputFiles\ARC_Input_Shields_Bathy.txt"
            LOG.warning('Moving forward with Default MIF Name: ' + MIF_Name)
            
        main(MIF_Name, self._args, self.quiet)

    def flood(self):
        with open(self._mifn, 'r') as file:
            lines = file.readlines()
            num_lines = len(lines)
            dem_file = get_parameter_name(lines, num_lines, 'DEM_File')
            strm_file = get_parameter_name(lines, num_lines, 'Stream_File')
            flow_file = get_parameter_name(lines, num_lines, 'COMID_Flow_File') or get_parameter_name(lines, num_lines, 'Comid_Flow_File') or get_parameter_name(lines, num_lines, 'Flow_File')
            vdt_database = get_parameter_name(lines, num_lines, 'Print_VDT_Database')
            curve_file = get_parameter_name(lines, num_lines, 'Print_Curve_File')

            q_fraction = float(get_parameter_name(lines, num_lines, 'Q_Limit') or 1.0)
            tw_factor = float(get_parameter_name(lines, num_lines, 'TopWidthDistanceFactor') or 1.5)
            flood_local = bool(get_parameter_name(lines, num_lines, 'Flood_Local')) or bool(get_parameter_name(lines, num_lines, 'LocalFloodOption'))
            ar_bathy_file = get_parameter_name(lines, num_lines, 'BATHY_Out_File')
            fs_bathy_file = get_parameter_name(lines, num_lines, 'FSOutBATHY')
            flood_impact_file = ''
            flood_map = get_parameter_name(lines, num_lines, 'OutFLD')

        if not dem_file:
            LOG.error('DEM file not found')
            return
        if not strm_file:
            LOG.error('Stream file not found')
            return
        if not flow_file:
            LOG.error('Flow file not found')
            return
        if not flood_map:
            LOG.error('Flood map file not found')
            return

        Curve2Flood_MainFunction(dem_file, strm_file, '', flow_file, curve_file, vdt_database, flood_map, flood_impact_file, q_fraction, 200, tw_factor, flood_local, 0.1, '', ar_bathy_file, fs_bathy_file, self.quiet)
        
    def set_log_level(self, log_level: str) -> 'Arc':
        handler = LOG.handlers[0]
        if log_level == 'debug':
            LOG.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        elif log_level == 'info':
            LOG.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)
        elif log_level == 'warn':
            LOG.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
        elif log_level == 'error':
            LOG.setLevel(logging.ERROR)
            handler.setLevel(logging.ERROR)
        else:
            LOG.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
            LOG.warning('Invalid log level. Defaulting to warning.')
            return
            
        LOG.info('Log Level set to ' + log_level)
        return self

def _main():
    parser = argparse.ArgumentParser(description='Run ARC')
    parser.add_argument('mifn', type=str, help='Model Input File Name')
    parser.add_argument('-l', '--log', type=str, help='Log Level', 
                        default='warn', choices=['debug', 'info', 'warn', 'error'])
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output progress bar and other non-error messages')
    parser.add_argument('-f', '--flood', action='store_true', help='Run flood mapper')
    args = parser.parse_args()
    arc = Arc(args.mifn, args.quiet)
    arc.set_log_level(args.log)
    
    if args.flood:
        arc.flood()
    else:
        arc.run()

    LOG.info('Finished')

    
    
if __name__ == "__main__":
    _main()
