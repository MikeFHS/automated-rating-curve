from __future__ import annotations
import argparse
import logging
from typing import Literal
from datetime import datetime

from arc import LOG
from arc.Automated_Rating_Curve_Generator import main

__all__ = ['Arc']

class Arc():
    """
    High-level ARC runner.

    This class provides a lightweight interface for running ARC from Python and
    is used by the ``arc`` console script.
    """
    _mifn: str = ""
    _args: dict = {}
    
    def __init__(self, mifn: str = "", args: dict | None = None, quiet: bool = False, processes: int | Literal["auto"] = 1) -> None:
        """Initialize an `Arc` instance.
        
        Parameters
        ----------
        mifn : str, optional
            Path to an ARC model input file (MIF).
        args : dict or None, optional
            Dictionary of key-value pairs corresponding to ARC input-file arguments. Will only be used if `mifn` is not provided.
        quiet : bool, optional
            If True, suppress progress bars and non-error log output.
        processes : int or {"auto"}, optional
            Number of worker processes for the per-stream-cell computation. Use ``"auto"`` to select serial vs. parallel based on domain size.
        
        Returns
        -------
        None
        """
        self._mifn = mifn
        self._args = args or {}
        self._quiet = quiet
        self._processes = processes
        if quiet:
            self.set_log_level('error')
        
    def run(self):
        """
        Run ARC.

        Returns
        -------
        None
            Outputs are written to disk based on input-file arguments.
        """
        LOG.info('Inputs to the Program is a Main Input File')
        LOG.info('\nFor Example:')
        LOG.info('  python Automated_Rating_Curve_Generator.py ARC_InputFiles/ARC_Input_File.txt')
        
        ### User-Defined Main Input File ###
        if self._mifn or self._args:
            LOG.info(f'Main Input File Given: {self._mifn}')
        else:
            #Read Main Input File
            MIF_Name = 'ARC_InputFiles/ARC_Input_File.txt'
            MIF_Name = r"C:\Projects\2024_FHS_FloodForecasting\ARC_Shields_Nencarta\nencarta_test_wsebathy_clean\yellowstone_wsebathy_clean\ARC_InputFiles\ARC_Input_Shields_Bathy.txt"
            LOG.warning('Moving forward with Default MIF Name: ' + MIF_Name)
            
        main(MIF_Name, self._args, self._quiet, self._processes)
        
    def set_log_level(self, log_level: str) -> 'Arc':
        """
        Set ARC's logging verbosity.

        Parameters
        ----------
        log_level : {"debug", "info", "warn", "error"}
            Desired log level.

        Returns
        -------
        Arc
            Self, for chaining.
        """
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
    """Command-line entry point for the ``arc`` console script."""
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
