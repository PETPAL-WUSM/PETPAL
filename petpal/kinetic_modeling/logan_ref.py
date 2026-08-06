"""Regional kinetic modeling with Logan (reference region)"""
import numba
import numpy as np

from petpal.kinetic_modeling.graphical_analysis import (cumulative_trapezoidal_integral,
                                                        get_index_from_threshold,
                                                        linear_least_squares_fit_with_stats)
from petpal.utils.time_activity_curve import TimeActivityCurve
from .kinetic_model_base import ModelConfig
from ..meta.auto_cli import auto_cli


@numba.njit
def logan_ref_region_solver(times: np.ndarray,
                            reference_activity: np.ndarray,
                            region_activity: np.ndarray,
                            k2_prime: float,
                            start_time: float,
                            end_time: float=600) -> tuple[float, float, float, float, float]:
    """
    Performs Logan with reference region input function on given input TAC, regional TAC, times,
    threshold, and population averaged reference region k2.

    Args:
        times (np.ndarray): Array of times in minutes.
        reference_activity (np.ndarray): Array of input TAC values
        region_activity (np.ndarray): Array of ROI TAC values
        k2_prime (float): Population averaged k2 value for the reference region.
        start_time (np.ndarray): Time point (in minutes) to begin integration.
        end_time (np.ndarray): Time point (in minutes) to end integration. Default 600.

    Returns:
        tuple: (slope, intercept, :math:`R^2`, slope standard error, intercept standard error)

    .. important::
        * The interpretation of the values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.

    """

    non_zero_indices = np.argwhere(region_activity != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    start_index = get_index_from_threshold(times_in_minutes=times[non_zero_indices],
                                        t_thresh_in_minutes=start_time)

    end_index = get_index_from_threshold(times_in_minutes=times[non_zero_indices],
                                        t_thresh_in_minutes=end_time)

    if len(times[non_zero_indices][start_index:end_index]) <= 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    logan_x = cumulative_trapezoidal_integral(xdata=times, ydata=reference_activity)
    logan_y = cumulative_trapezoidal_integral(xdata=times, ydata=region_activity)

    logan_x_ref_region_term = reference_activity[non_zero_indices][start_index:end_index]/k2_prime
    logan_x_numerator = logan_x[non_zero_indices][start_index:end_index] + logan_x_ref_region_term
    logan_denominator = region_activity[non_zero_indices][start_index:end_index]
    logan_x = logan_x_numerator / logan_denominator
    logan_y = logan_y[non_zero_indices][start_index:end_index] / logan_denominator

    logan_values = linear_least_squares_fit_with_stats(xdata=logan_x, ydata=logan_y)

    return logan_values


class LoganRefConfig(ModelConfig):
    """Config settings for logan reference tissue"""
    def __init__(self):
        super().__init__(model_name='LoganRef',
                         model_solver=logan_ref_region_solver,
                         required_pars=["k2_prime","start_time","end_time"],
                         fitted_pars=['DVR','Intercept','RSquared', 'SE_DVR', 'SE_intercept','BP'])


    def run_model(self,
                  reference_tac: TimeActivityCurve,
                  region_tac: TimeActivityCurve):
        """Run logan reference"""
        fits = self.model_solver(times=reference_tac.times,
                                 reference_activity=reference_tac.activity,
                                 region_activity=region_tac.activity,
                                 k2_prime=self.model_pars.k2_prime,
                                 start_time=self.model_pars.start_time,
                                 end_time=self.model_pars.end_time)
        bp = fits[0] - 1
        fit_result = [*fits, bp]
        return fit_result

    def set_tacs_data(self,
                       tacs_path: str,
                       reference_region: str):
        self.tacs = self.tacs_loader.load(tacs_path=tacs_path)
        self.reference_tac = self.tacs[reference_region]

    def __call__(self,
                 reference_region: str,
                 regional_tacs_path: str,
                 save_path: str,
                 k2_prime: float,
                 start_time: float,
                 end_time: float=600):
        """
        Fit all regions with Logan reference kinetic model.

        Args:
            
            reference_region (str): Label for the reference region in the TACs spreadsheet.
            regional_tacs_path (str): Path to TACs spreadsheet.
            save_path (str): Path to where modeling parameters are saved.
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
            start_time (np.ndarray): Time point (in minutes) to begin logan model integration.
            end_time (np.ndarray): Time point (in minutes) to end logan model integration. Default
                600.
        """
        self.set_required_pars(k2_prime=k2_prime, start_time=start_time, end_time=end_time)
        self.set_tacs_data(tacs_path=regional_tacs_path, reference_region=reference_region)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)


def main():
    auto_cli(petpal_class=LoganRefConfig)

if __name__=='__main__':
    main()
