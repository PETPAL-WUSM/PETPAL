"""Regional kinetic modeling with Logan (reference region)"""
import numba
import numpy as np

from petpal.kinetic_modeling import graphical_analysis
from petpal.kinetic_modeling.graphical_analysis import (cumulative_trapezoidal_integral,
                                                        get_index_from_threshold,
                                                        linear_least_squares_fit_with_stats)
from petpal.utils.time_activity_curve import TimeActivityCurve
from .kinetic_model_base import ModelConfig
from ..meta.auto_cli import auto_cli


@numba.njit
def logan_ref_region_solver(tac_times_in_minutes: np.ndarray,
                            input_tac_values: np.ndarray,
                            region_tac_values: np.ndarray,
                            k2_prime: float,
                            start_time: float,
                            end_time: float=600) -> tuple[float, float, float]:
    """
    Performs Logan with reference region input function on given input TAC, regional TAC, times,
    threshold, and population averaged reference region k2.

    Args:
        tac_times_in_minutes (np.ndarray): Array of times in minutes.
        input_tac_values (np.ndarray): Array of input TAC values
        region_tac_values (np.ndarray): Array of ROI TAC values
        k2_prime (float): Population averaged k2 value for the reference region.
        start_time (np.ndarray): Time point (in minutes) to begin integration.
        end_time (np.ndarray): Time point (in minutes) to end integration. Default 600.

    Returns:
        tuple: (slope, intercept, :math:`R^2`, slope standard error, intercept standard error)

    .. important::
        * The interpretation of the values depends on the underlying kinetic model.
        * We assume that the input TAC and ROI TAC values are sampled at the same times.

    """

    non_zero_indices = np.argwhere(region_tac_values != 0.).T[0]

    if len(non_zero_indices) <= 2:
        return np.nan, np.nan, np.nan

    start_index = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=start_time)

    end_index = get_index_from_threshold(times_in_minutes=tac_times_in_minutes[non_zero_indices],
                                        t_thresh_in_minutes=end_time)

    if len(tac_times_in_minutes[non_zero_indices][start_index:end_index]) <= 2:
        return np.nan, np.nan, np.nan

    logan_x = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=input_tac_values)
    logan_y = cumulative_trapezoidal_integral(xdata=tac_times_in_minutes, ydata=region_tac_values)

    logan_x_ref_region_term = input_tac_values[non_zero_indices][start_index:end_index]/k2_prime
    logan_x_numerator = logan_x[non_zero_indices][start_index:end_index] + logan_x_ref_region_term
    logan_denominator = region_tac_values[non_zero_indices][start_index:end_index]
    logan_x = logan_x_numerator / logan_denominator
    logan_y = logan_y[non_zero_indices][start_index:end_index] / logan_denominator

    logan_values = linear_least_squares_fit_with_stats(xdata=logan_x, ydata=logan_y)

    return logan_values


class LoganRefConfig(ModelConfig):
    """Config settings for logan reference tissue"""
    def __init__(self):
        super().__init__(model_name='LoganRef',
                         model_solver=graphical_analysis.logan_ref_region_analysis_with_rsquared,
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

    def set_tacs_data(self,
                       tacs_path: str,
                       reference_region: str):
        self.tacs = self.tacs_loader.load(tacs_path=tacs_path)
        self.reference_tac = self.tacs[reference_region]

    def __call__(self,
                 reference_region: str,
                 regional_tacs_path: str,
                 save_path: str,
                 t_star: float,
                 k2_prime: float):
        """
        Fit all regions with Logan reference kinetic model.

        Args:
            
            reference_region (str): Label for the reference region in the TACs spreadsheet.
            regional_tacs_path (str): Path to TACs spreadsheet.
            save_path (str): Path to where modeling parameters are saved.
            t_star (str): Beginning model time for Logan reference.
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
        """
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        self.set_tacs_data(tacs_path=regional_tacs_path, reference_region=reference_region)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)


def main():
    auto_cli(petpal_class=LoganRefConfig)

if __name__=='__main__':
    main()
