"""Wrappers for generating CLI functions from PETPAL callable objects"""
import argparse
import logging
import inspect
import sys
import copy
from pydoc import locate


def camel_to_kebab_case(name: str) -> str:
    """Convert a string in camel case to kebab case.
    
    Camel case is used for class objects in Python, e.g. MyClass.
    Kebab case is used for programs, such as my-prog.
    This function converts a string in camel case to kebab case, for the purpose of assigning
    CLI program names to callable class objects.
    
    Args:
        name (str): The camel case phrase to be converted to kebab case.
    
    Returns:
        kebab_name (str): Input phrase converted to kebab case.
    """
    if len(name)==1:
        return name.lower()

    kebab_name = name[0].lower()
    for char in name[1:]:
        if char.isupper():
            kebab_name += f'-{char.lower()}'
        else:
            kebab_name += char
    return kebab_name


def args_kwargs_to_dictionary(args: argparse.Namespace) -> dict:
    """Convert Namespace to dictionary and reassign keyword arguments to dictionary values.
    
    Args:
        args (argparse.Namespace): The args resulting from parsing command line arguments.
    
    Returns:
        call_eval (dict): Dictionary with arguments and keyword arguments from inspecting the class
            __call__ function.
    """
    arg_vals = vars(args)
    call_eval = arg_vals.copy()

    for arg in arg_vals:
        value = arg_vals[arg]
        if isinstance(value, dict):
            for kwarg in value:
                call_eval[kwarg] = value[kwarg]
            call_eval.pop(arg)

    return call_eval


class ParseKwargs(argparse.Action):
    """Action to parse keyword arguments."""
    SUPPORTED_KWARG_TYPES = [str, float, int, bool]

    def __call__(self, parser, namespace, values, option_string=None):
        """Creates a dictionary within the args namespace holding keyword arguments.
        
        Kwargs are used like so: --kwargs int:frame=4
        Each kwarg must specify type before a colon, followed by the argument name, an equals sign
        to delimit the argument value. Kwargs are currently limited to a small number of standard
        types: str, float, int, bool."""
        setattr(namespace, self.dest, {})
        for value in values:
            kwarg_type, kwarg_pair = value.split(':')
            kwarg_locator = locate(kwarg_type)
            kwarg_name, kwarg_value = kwarg_pair.split('=')

            if kwarg_locator not in self.SUPPORTED_KWARG_TYPES:
                raise TypeError(f"CLI Kwargs only supports types: {self.SUPPORTED_KWARG_TYPES}."
                                f"Got {kwarg_locator}.")

            getattr(namespace, self.dest)[kwarg_name] = kwarg_locator(kwarg_value)


class PetpalLogging:
    """Tool for logging CLI commands.
    
    Records PETPAL CLI commands in a log file. Creates a new logfile at the provided path, if one 
    does not already exist. If it points to an existing file, any logging is added to the bottom of
    the file. Logs command line arguments only."""
    def __init__(self, logfile: str=None):
        if logfile is not None:
            logging.basicConfig(filename=logfile, level=logging.INFO, format='')
            self.logger = logging.getLogger('')
            self.cli_args = copy.deepcopy(sys.argv)
            self.log_cli_args()

    def log_cli_args(self):
        """Log the command line that was used to run PETPAL.
        
        Records the literal PETPAL command line tool, and any interpolated variables as they are
        interpreted by Python.
        """
        self.logger.info(' '.join(self.cli_args))


def type_identifier(arg_type_name: str) -> tuple:
    """Identify the locator, number of arguments, and default value for the argument type."""
    arg_default = None
    arg_split_default = arg_type_name.split(' = ')
    if len(arg_split_default)>1:
        arg_default = arg_split_default[1]

    arg_type = locate(arg_split_default[0])
    nargs = 1
    arg_split_union = arg_split_default[0].split(' | ')
    ntypes = len(arg_split_union)
    for atype in ntypes:
        if 'tuple' or 'list' in atype:
            nargs = '+'

    return arg_type, nargs, arg_default


def auto_cli(petpal_class: object):
    """Generate a command line interface for a PETPAL function
    
    Args:
        petpal_class (object): Class defined in PETPAL that can be instantiated without specifying
            __init__ arguments. Must contain function __call__ with a docstring.

    Important:
        The provided class must be designed such that any __init__ arguments can be set with
        defaults, and those defaults should only be changed for testing and development purposes.
        Otherwise, it would be impossible to tell which CLI arguments should be assigned to the
        __init__ function and which should be assigned to __call__.

        Additionally, any __call__ arguments and keyword arguments must be able to be specified from
        the command line, such as strings, integers, or floats.

        Keyword arguments are only intended to be used with the CLI when it would be excessive and
        cumbersome to implement all possible arguments of a function. Any expected user inputs
        should be specified, documented, and type hinted in the __call__ args, reserve kwargs only
        for providing user flexibility where necessary, as it carries the risk of unintended
        behavior.

    Example:
        
        .. code-block:: python

            import numpy as np
            from petpal.meta.auto_cli import auto_cli
            import external_func

            class MyClass:
                mri_img: ants.ANTsImage
                pet_img: ants.ANTsImage

                def my_func(self, **kwargs):
                    return external_func(self.mri_img, self.pet_img, **kwargs)

                def __call__(self, mri_img_path, pet_img_path, out_img_path, **kwargs):
                    self.mri_img = ants.image_read(mri_img_path)
                    self.pet_img = ants.image_read(pet_img_path)
                    output_img = my_func(**kwargs)
                    ants.image_write(output_img, output_img)
            
            def main():
                # Creates CLI for class my_class
                # __call__ args are interpreted as required arguments
                # **kwargs is interpreted as optional keyword arguments
                # usage: python my_class.py my-class --mri-img-path [MRI_IMG_PATH] \\
                #          --pet-img-path [PET_IMG_PATH] --kwargs [kwarg1=val1 kwarg2=val2 ...]
                # Add file path to pyproject.toml to generate a command shortcut

                auto_cli(petpal_class=my_class)

            if __name__=='__main__':
                main()
    """
    parser = argparse.ArgumentParser(prog=petpal_class.__name__,
                                     description=petpal_class.__call__.__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    call_signature = inspect.signature(petpal_class.__call__)
    call_parameters = call_signature.parameters.items()
    for _, call_parameter in call_parameters:
        arg_name = str(call_parameter)
        if arg_name=='self':
            continue
        arg_and_type = arg_name.split(': ')
        print(arg_and_type)
        if len(arg_and_type)==2:
            arg_name = f'--{arg_and_type[0]}'.replace('_','-')
            arg_type, nargs, arg_default = type_identifier(arg_type_name=arg_and_type[1])
            print(arg_type)
            if nargs==1:
                parser.add_argument(arg_name,type=arg_type,required=True,default=arg_default)
            else:
                parser.add_argument(arg_name,type=arg_type,required=True,nargs=nargs,default=arg_default)
        elif arg_and_type[0].startswith('**'):
            kwarg_name = arg_and_type[0].replace('**','--').replace('_','-')
            parser.add_argument(kwarg_name, nargs='*', action=ParseKwargs, required=False)
    parser.add_argument('--logfile',type=str,default=None,required=False)

    args = parser.parse_args()
    arg_vals = args_kwargs_to_dictionary(args=args)
    print(arg_vals)

    logger = PetpalLogging(logfile=arg_vals['logfile'])
    arg_vals.pop('logfile')

    init_class = petpal_class()
    init_class(**arg_vals)
