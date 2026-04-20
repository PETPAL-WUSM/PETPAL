"""Regional kinetic modeling with Logan (reference region)"""
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Sequence, Protocol, Dict, Type, Any, Optional, Callable
import tempfile
import os
import dataclasses
import numpy as np
import pandas as pd
import ants
from petpal.kinetic_modeling import graphical_analysis, reference_tissue_models
from petpal.utils.time_activity_curve import TimeActivityCurve
from petpal.utils.scan_timing import ScanTimingInfo
from petpal.io.table import TableSaver, RegionalTacsLoader
from petpal.utils.dimension import gen_3d_img_from_timeseries
from ..kinetic_model_base import ModelConfig
from ...meta.auto_cli import auto_cli

class LoganRefConfig(ModelConfig):
    """Config settings for logan reference tissue"""
    def __init__(self):
        super().__init__(model_solver=graphical_analysis.logan_ref_region_analysis_with_rsquared,
                         required_pars=["t_star","k2_prime"],
                         fitted_pars=['DVR','Intercept','RSquared','BP'])


    def run_model(self,
                  reference_tac: TimeActivityCurve,
                  region_tac: TimeActivityCurve):
        """Run logan reference"""
        fits = self.model_solver(tac_times_in_minutes=reference_tac.times,
                        input_tac_values=reference_tac.activity,
                        region_tac_values=region_tac.activity,
                        t_thresh_in_minutes=self.model_pars.t_star,
                        k2_prime=self.model_pars.k2_prime)
        bp = fits[0] - 1
        fit_result = [*fits, bp]
        return fit_result

    def __call__(self,
                 reference_region,
                 regional_tacs_path,
                 save_path,
                 t_star: float,
                 k2_prime: float):
        self.tacs = self.tacs_loader.load(tacs_path=regional_tacs_path)
        self.reference_tac = self.tacs[reference_region]
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)


def main():
    auto_cli(petpal_class=LoganRefConfig)

if __name__=='__main__':
    main()
