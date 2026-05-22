"""Regional kinetic modeling with Logan (reference region)"""
from petpal.kinetic_modeling import graphical_analysis
from petpal.utils.time_activity_curve import TimeActivityCurve
from .kinetic_model_base import ModelConfig
from ..meta.auto_cli import auto_cli

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
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)


def main():
    auto_cli(petpal_class=LoganRefConfig)

if __name__=='__main__':
    main()
