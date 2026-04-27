from __future__ import annotations
import argparse
import logging
from typing import Literal
from datetime import datetime

from arc import LOG
from arc.Automated_Rating_Curve_Generator import main, get_parameter_name

__all__ = ['Arc']

class Arc():
    _mifn: str = ""
    _args: dict = {}
    
    def __init__(self, mifn: str = "", args: dict | None = None, quiet: bool = False, processes: int | Literal["auto"] = 1) -> None:
        self._mifn = mifn
        self._args = args or {}
        self.quiet = quiet
        self.processes = processes
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
            
        main(MIF_Name, self._args, self.quiet, self.processes)
        
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
    parser.add_argument('-p', '--processes', type=str, default='1', help='Number of worker processes, or \"auto\"')
    args = parser.parse_args()
    processes: int | Literal["auto"]
    processes = "auto" if args.processes.strip().lower() == "auto" else int(args.processes)
    arc = Arc(args.mifn, args=None, quiet=args.quiet, processes=processes)
    arc.set_log_level(args.log)
    arc.run()

    LOG.info('Finished')

    
    
if __name__ == "__main__":
    _main()
