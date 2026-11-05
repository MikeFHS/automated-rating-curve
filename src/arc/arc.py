from __future__ import annotations

import argparse
import logging

from arc import LOG
from arc.config import Configs
from arc.pipeline import Results, run

__all__ = ['Arc']

class Arc:
    """
    Run ARC from Python, as the ``arc`` command does.

    Give it an input file or a dictionary of the same keys, then call `run`. See the Model Input File page of
    the documentation for the keys.
    """
    def __init__(self, mifn: str = "", args: dict | None = None, quiet: bool = False) -> None:
        """
        Parameters
        ----------
        mifn : str, optional
            Path to an ARC model input file (MIF): YAML, or text with a key and its value on each line.
        args : dict, optional
            The input file's keys and values, used only if ``mifn`` isn't given.
        quiet : bool
            Hide the progress bars and every log message but errors.
        """
        self._mifn = mifn
        self._args = args or {}
        self._quiet = quiet
        if quiet:
            self.set_log_level('error')

    def run(self) -> Results:
        """
        Run ARC, writing the outputs the inputs name.

        Returns
        -------
        Results
            What the run made (see `arc.pipeline.Results`).
        """
        if self._mifn:
            LOG.info(f'Main Input File Given: {self._mifn}')
            configs = Configs.from_file(self._mifn)
        elif self._args:
            configs = Configs.from_mapping(self._args)
        else:
            raise ValueError('Arc needs a model input file (mifn) or a dictionary of input arguments (args).')
        
        return run(configs, quiet=self._quiet)

    def set_log_level(self, log_level: str) -> Arc:
        """
        Set ARC's logging verbosity.

        Parameters
        ----------
        log_level : str
            ``"debug"``, ``"info"``, ``"warn"`` or ``"error"``. Anything else sets ``"warn"``, with a warning.

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
            return self

        LOG.info('Log Level set to ' + log_level)
        return self

def _main():
    """Command-line entry point for the ``arc`` console script."""
    parser = argparse.ArgumentParser(description='Run ARC')
    parser.add_argument('mifn', type=str, help='Model Input File Name')
    parser.add_argument('-l', '--log', type=str, help='Log level (default warn, or error with --quiet)',
                        default=None, choices=['debug', 'info', 'warn', 'error'])
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output progress bar and other non-error messages')
    args = parser.parse_args()
    arc = Arc(args.mifn, args=None, quiet=args.quiet)
    if args.log or not args.quiet:  # --quiet keeps the errors alone unless a level is given
        arc.set_log_level(args.log or 'warn')
    arc.run()

    LOG.info('Finished')



if __name__ == "__main__":
    _main()
