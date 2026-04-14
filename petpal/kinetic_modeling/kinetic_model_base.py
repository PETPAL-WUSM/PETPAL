"""Base classes for kinetic analysis"""

class ModelConfig:
    r"""
    Base class for config settings to apply to kinetic models
    """
    def __init__(self, model_solver: Callable, required_pars: list[str], fitted_pars: list[str]):
        r"""
        Initialize a TCM model configuration.
        """
        self.model_solver = model_solver
        self.required_pars = required_pars
        self.fitted_pars = fitted_pars
        self.num_params = len(required_pars)
        self.model_pars = None

    def required_parameter_setter(self, **pars):
        """Set each parameter in required_pars"""
        model_pars = namedtuple('Pars',self.required_pars)
        self.model_pars = model_pars(**pars)

    def null_result(self) -> np.ndarray:
        """Return a numpy array of NaNs indexed by the model's fitted_pars."""
        n_fitted_pars = len(self.fitted_pars)
        fits = np.full(n_fitted_pars,np.nan)
        return fits

    @abstractmethod
    def run_model(self, reference_tac, region_tac) -> Sequence[float]:
        ...

    @staticmethod
    def normalize_name(name: str) -> str:
        r"""
        Normalize a model name to a standard format.

        Converts the name to lowercase and replaces spaces and underscores with hyphens
        for consistent model name lookup.

        Args:
            name (str): The model name to normalize.

        Returns:
            str: The normalized model name.

        Example:
            .. code-block:: python

                from petpal.kinetic_modeling.tac_fitting import TcmModelConfig
                TcmModelConfig.normalize_name("Serial 2TCM")  # Returns "serial-2tcm"
                TcmModelConfig.normalize_name("1TCM")  # Returns "1tcm"
        """
        return name.lower().replace(' ', '_').replace('_', '-')

    def fit_regions(self) -> pd.DataFrame:
        """Run the kinetic model on all of the regions"""
        fit_results = pd.DataFrame(index=self.model_config.fitted_pars)
        tacs = self.tacs
        for region,tac in tacs.items():
            try:
                region_fit = self.run_model(reference_tac=self.reference_tac,
                                            region_tac=tac)
            except Exception:
                region_fit = self.model_config.null_result()
            fit_results[region] = region_fit
        return fit_results

    def __call__(self,
                 input_tac_path,
                 regional_tacs_path,
                 save_path,
                 **run_kwargs):
        self.tacs = self.tacs_loader.load_tacs_sheet(tacs_path=regional_tacs_path)
        self.reference_tac = TimeActivityCurve.from_tsv(filename=input_tac_path)
        self.model_config.set_required_pars(**run_kwargs)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)



class LoganRefConfig(ModelConfig):
    """Config settings for logan reference tissue"""
    def __init__(self):
        super().__init__(model_solver=graphical_analysis.logan_ref_region_analysis_with_rsquared,
                         required_pars=["t_star","k2_prime"],
                         fitted_pars=['DVR','Intercept','R-squared','BP'])


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



class KineticModeling:
    """Interface for running a kinetic model on TACs for each region"""
    def __init__(self,
                 tacs_loader: Optional[RegionalTacsLoader] = None,
                 table_saver: Optional[TableSaver] = None,):
        self.tacs = pd.DataFrame()
        self.input_tac: TimeActivityCurve = None
        self._model_factory = model_factory or reference_model_factory
        self.table_saver = table_saver or TableSaver()
        self.tacs_loader = tacs_loader or RegionalTacsLoader()
        self.model_config: BaseModelConfig | None = None

    def set_model(self, model_name: str):
        """Set the model to run"""
        self.model_config = self._model_factory(model_name)

    def run_model(self,tac):
        """Run the kinetic model"""
        model_fit = self.model_config.run_model(input_tac=self.input_tac,
                                                region_tac=tac)
        return model_fit


    def fit_regions(self) -> pd.DataFrame:
        """Run the kinetic model on all of the regions"""
        fit_results = pd.DataFrame(index=self.model_config.fitted_pars)
        tacs = self.tacs
        for region,tac in tacs.items():
            try:
                region_fit = self.run_model(tac=tac)
            except Exception:
                region_fit = self.model_config.null_result()
            fit_results[region] = region_fit
        return fit_results

    def __call__(self,
                 input_tac_path,
                 regional_tacs_path,
                 save_path,
                 **run_kwargs):
        self.tacs = self.tacs_loader.load_tacs_sheet(tacs_path=regional_tacs_path)
        self.input_tac = TimeActivityCurve.from_tsv(filename=input_tac_path)
        self.model_config.set_required_pars(**run_kwargs)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)