"""Solve MRTM2 regional analysis."""
import numba
import numpy as np

from petpal.kinetic_modeling.graphical_analysis import (cumulative_trapezoidal_integral,
                                                        get_index_from_threshold)
from petpal.utils.time_activity_curve import TimeActivityCurve
from petpal.kinetic_modeling.kinetic_model_base import ModelConfig
from petpal.meta.auto_cli import auto_cli
from petpal.kinetic_modeling.reference_tissue_models import calc_bp_from_mrtm2_2003_fit


@numba.njit(fastmath=True)
def mrtm2_solver(times: np.ndarray,
                 reference_activity: np.ndarray,
                 region_activity: np.ndarray,
                 k2_prime: float,
                 start_time: float,
                 end_time: float=600,
                 uncertainty: np.ndarray=None):
    r"""Solves MRTM2 kinetic model based on reference region activity and activity in a region of
    interest.

    .. important::
        This function assumes that both TACs are sampled at the same time, and that the time is in
        minutes.
    
    This method solves for coefficients :math:`-\frac{V}{V^{\prime}b}` and :math:`\frac{1}{b}`
    based on the multilinear regression model:

    .. math::

        C(T) = -\frac{V}{V^{\prime}b}\left(\int_{0}^{T}C^{\prime}(t)\mathrm{d}t
        -\frac{1}{k_{2}^{\prime}}C^{\prime}(T) \right)
        + \frac{1}{b} \int_{0}^{T}C(t)\mathrm{d}t

    Args:
        reference_activity (np.ndarray): Activity in the reference region for each frame.
        region_activity (np.ndarrray): Activity in the region of interest for each frame. Units
            must match that of reference_activity.
        k2_prime (float): Kinetic parameter k2 for the reference region. Typically estimated based
            on an initial run of MRTM.
        start_time (float): Time measured in minutes from start of scan at which to begin model.
        end_time (float): Time measured in minutes from start of scan at which to end model.
            Default 600.
        uncertainty (float): Uncertainty for each frame in the PET scan. Not yet implemented.

    Returns:
        fit_ans (np.ndarray): Array of linear least square fits. Referred to in subsequent analysis
            as Mrtm2Coefficient1 and Mrtm2Coefficient2 for use in file names and tables.
    """
    if uncertainty is None:
        weights = np.ones_like(times)
    else:
        weights = np.ones_like(times) # TODO: effective calculation of frame weights

    start_index = get_index_from_threshold(times_in_minutes=times,
                                           t_thresh_in_minutes=start_time)
    if start_index == -1:
        return np.asarray([np.nan, np.nan])

    end_index = get_index_from_threshold(times_in_minutes=times,
                                         t_thresh_in_minutes=end_time)

    x1 = cumulative_trapezoidal_integral(xdata=times, ydata=reference_activity, initial=0.0)
    x1 += reference_activity / k2_prime
    x2 = cumulative_trapezoidal_integral(xdata=times, ydata=region_activity, initial=0.0)

    
    y = region_activity[start_index:end_index]*weights[start_index:end_index]
    x_matrix = np.ones((len(y), 2), float)
    x_matrix[:,0] = x1[start_index:end_index]*weights[start_index:end_index]
    x_matrix[:,1] = x2[start_index:end_index]*weights[start_index:end_index]

    fit_ans = np.linalg.lstsq(x_matrix, y)[0]
    return fit_ans


class Mrtm2Config(ModelConfig):
    """Config settings for MRTM2. Initialize and call class to run model on all regions."""
    def __init__(self):
        super().__init__(model_name="Mrtm2",
                         model_solver=mrtm2_solver,
                         required_pars=["k2_prime","start_time","end_time"],
                         fitted_pars=["Mrtm2Coefficient1","Mrtm2Coefficient2","BP"])


    def run_model(self,
                  reference_tac: TimeActivityCurve,
                  region_tac: TimeActivityCurve):
        """Run MRTM2 model on a single region.
        
        Args:
            reference_tac (TimeActivityCurve): TAC for the reference region.
            region_tac (TimeActivityCurve): TAC for the region to model with MRTM2.
    
        Returns:
            fit_result (list): List of fitted parameters output from MRTM2 solver with estimated
                binding potential (BP) appended to the end."""
        fits = self.model_solver(times=reference_tac.times,
                                 reference_activity=reference_tac.activity,
                                 region_activity=region_tac.activity,
                                 k2_prime=self.model_pars.k2_prime,
                                 start_time=self.model_pars.start_time,
                                 end_time=self.model_pars.end_time,
                                 uncertainty=region_tac.uncertainty)
        bp = calc_bp_from_mrtm2_2003_fit(fits)
        fit_result = [*fits, bp]
        return fit_result

    def __call__(self,
                 reference_region: str,
                 regional_tacs_path: str,
                 save_path: str,
                 k2_prime: float,
                 start_time: float,
                 end_time: float=600):
        """
        Fit all regions with MRTM2 kinetic model.

        Args:
            
            reference_region (str): Label for the reference region in the TACs spreadsheet.
            regional_tacs_path (str): Path to TACs spreadsheet.
            save_path (str): Path to where modeling parameters are saved.
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
            start_time (np.ndarray): Time point (in minutes) to begin MRTM2 model integration.
            end_time (np.ndarray): Time point (in minutes) to end MRTM2 model integration. Default
                600.
        """
        self.set_required_pars(k2_prime=k2_prime, start_time=start_time, end_time=end_time)
        self.set_tacs_data(tacs_path=regional_tacs_path, reference_region=reference_region)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)


def main():
    auto_cli(petpal_class=Mrtm2Config)

if __name__=='__main__':
    main()
