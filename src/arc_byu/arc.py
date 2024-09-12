import argparse
import logging
from datetime import datetime

from arc_byu import LOG
from arc_byu.Automated_Rating_Curve_Generator import main

class Arc():
    _mifn: str = ""
    
    def __init__(self, mifn: str = "") -> None:
        self._mifn = mifn
        
    def run(self):
        global starttime
        global MIF_Name
        starttime = datetime.now()
    
        LOG.info('Inputs to the Program is a Main Input File')
        LOG.info('\nFor Example:')
        LOG.info('  python Automated_Rating_Curve_Generator.py ARC_InputFiles/ARC_Input_File.txt')
        
        ### User-Defined Main Input File ###
        if self._mifn:
            MIF_Name = self._mifn
            LOG.info('Main Input File Given: ' + MIF_Name)
        else:
            #Read Main Input File
            MIF_Name = 'ARC_InputFiles/ARC_Input_File.txt'
            LOG.warning('Moving forward with Default MIF Name: ' + MIF_Name)
            
        main(MIF_Name)
        
    def set_log_level(self, log_level: str):
        if log_level == 'debug':
            LOG.setLevel(logging.DEBUG)
        elif log_level == 'info':
            LOG.setLevel(logging.INFO)
        elif log_level == 'warn':
            LOG.setLevel(logging.WARNING)
        elif log_level == 'error':
            LOG.setLevel(logging.ERROR)
        else:
            LOG.setLevel(logging.WARNING)
            LOG.warning('Invalid log level. Defaulting to warning.')
            return
            
        LOG.info('Log Level set to ' + log_level)

def _main():
    parser = argparse.ArgumentParser(description='Run ARC')
    parser.add_argument('mifn', type=str, help='Model Input File Name')
    parser.add_argument('-l', '--log', type=str, help='Log Level', 
                        default='warn', choices=['debug', 'info', 'warn', 'error'])
    args = parser.parse_args()
    arc = Arc(args.mifn)
    arc.set_log_level(args.log)
    
    arc.run()
    
if __name__ == "__main__":
    _main()