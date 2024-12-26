import logging
from typing import Union, List, Tuple, Dict

import casadi as ca
from casadi.tools import struct_symSX, entry


class NotDefinedParameter:
    """ Helper class to store a single parameter and their possible types"""

    def __init__(self, description: str, types: Union[type, List[type]]):
        self.description = description
        if type(types) == list:
            self.types = types
        else:
            self.types = [types]

    def __str__(self):
        return "Unspecified Parameter of type(s) " + str(self.types)

    def __repr__(self):
        return self.__str__()


class SymbolicParameter:
    FIX = 'fix'
    FREE = 'free'

    def __init__(self, name: str, type: str, defaultValue, **kwargs):
        self.name = name
        self.defaultValue = defaultValue
        self.entry = entry(self.name, **kwargs)

        assert type in [self.FIX, self.FREE], f"Type has to be either {self.FIX} or {self.FREE}"
        self.type = type


class FixedSymbolicParameter(SymbolicParameter):
    def __init__(self, name: str, defaultValue, **kwargs):
        super().__init__(name, SymbolicParameter.FIX, defaultValue, **kwargs)


class FreeSymbolicParameter(SymbolicParameter):
    def __init__(self, name: str, defaultValue, **kwargs):
        super().__init__(name, SymbolicParameter.FREE, defaultValue, **kwargs)


class ParametersBaseClass:
    """
    A class to store the parameters of a model. The parameters are stored as attributes of the class.

    1) The class also supports marking some of the attributes as symbolic variables using the function `declareSyms()`.
    The symbolic variables can either be:
        - fixed symbolic variables, which are not optimized, but are used to define the model, can be passed to the solver as parameter
        - free symbolic variables, which are optimized by the solver

    2) The symbolic variables can be replaced by different symbolic values (of the same structure)
    using the function `replaceFreeSyms()`. This is useful if the optimization variables are predefined in a structure.

    3) After optimization, the optimal values of the (fixed and free) symbolic variables
    can be added to the class using the function `addSymsOptimalValues()`.

    The symbolic variables are also available as a) attributes of the class and b) as a list of casadi SX.
    This is useful for the creation of casadi functions that take in the symbolic variables as input.

    For example:

    >>> params = Parameters()
    >>> f = ca.Function('f', [x,u] + self.syms_list, [xdot], ['x','u'] + params.syms_list_names)
    >>> f(x0, u0, *params.syms_list)
    """

    class StateEnum:
        INITIALIZED = 'initialized'
        SYMS_DECLARED = 'syms_declared'
        SYMS_REPLACED = 'syms_replaced'
        OPTIMAL_VALUES_ADDED = 'optimal_values_added'

    def __init__(self, **kwargs):
        # internal dictionary to store the types of the parameters
        self._typesDict = {}
        self._descriptionDict = {}

        # symbolic variables
        self._syms_free_struct: ca.tools.struct = struct_symSX([])
        self._syms_fix_struct: ca.tools.struct = struct_symSX([])

        # optimal symbolic variables
        self._syms_free_struct_opt: ca.tools.struct = None
        self._syms_fix_struct_opt: ca.tools.struct = None

        # symbolic variables instance list
        self._syms_dict_instance_fix: Dict[str, SymbolicParameter] = {}
        self._syms_dict_instance_free: Dict[str, SymbolicParameter] = {}

        self._optimized_syms_names: List[str] = []
        # self.syms_initial: ca.tools.struct = struct_symSX([])(0)

        self.state = self.StateEnum.INITIALIZED

    @property
    def syms_free_struct(self) -> ca.tools.struct:
        """ The symbolic (free) variables as a casadi struct"""
        return self._syms_free_struct

    @property
    def syms_fix_struct(self) -> ca.tools.struct:
        """ The symbolic (fixed) variables as a casadi struct"""
        return self._syms_fix_struct

    @property
    def syms_list(self) -> List[ca.SX]:
        """
        A list of the symbolic variables, read from the internal struct. Useful for the creation of casadi functions
        that take in the symbolic variables as input.
        """
        return_list = []

        # iterate over both the FIXED and OPTIMIZATION parameter structs:
        for symsstruct in [self._syms_fix_struct, self._syms_free_struct]:
            # iterate over the entries of the struct
            for key in symsstruct.keys():
                _entry = symsstruct[key]

                # check if the entry is a list or a casadi SX
                assert type(_entry) in [list,
                                        ca.SX], f"The symbolic struct entry {key} has to be a list[SX] or a casadi SX but is of type {type(_entry)}"
                if isinstance(_entry, list):
                    # unpack the list
                    for element in _entry:
                        assert type(element) == ca.SX, "The entries of the list have to be casadi SX"
                        return_list.append(element)
                else:
                    return_list.append(_entry)

        return return_list

    @property
    def syms_opt_all_list(self) -> List[ca.DM]:
        """
        A list of the optimal symbolic variables. This is useful if a casadi function that was created with the
        symbolic variables as input now, after optimization, requires the optimal values.
        """
        assert self.state == self.StateEnum.OPTIMAL_VALUES_ADDED, "The optimal values have not been added to the class, call addSymsOptValue to set them."
        assert self._syms_free_struct_opt is not None, "The optimal symbolic variables have not been set, call addSymsOptValue to set them."
        return_list = []

        # iterate over the entries of the struct
        for symstruct in [self._syms_fix_struct_opt, self._syms_free_struct_opt]:
            for key in symstruct.keys():
                entry = symstruct[key]

                # check if the entry is a list or a casadi SX
                assert type(entry) in [list,
                                       ca.DM], f"The optimal symbolic struct has to be a list[DM] or a casadi DM, but is {type(entry)} "
                if isinstance(entry, list):
                    # unpack the list
                    for element in entry:
                        assert type(
                            element) == ca.DM, f"The entries of the list have to be casadi DM, but is {type(element)}"
                        return_list.append(element)
                else:
                    return_list.append(entry)
        return return_list

    @property
    def syms_list_names(self) -> List[str]:
        """
        A list of the names of the symbolic variables, the names are prefixed with 'p_' to indicate that they are parameters.
        Useful for the creation of casadi functions that require the names of the parameters.

        :return: A list of strings with the names of the symbolic variables.
        """
        return_list = []

        # iterate over the entries of the struct
        for symsstruct, prefix in zip([self._syms_fix_struct, self._syms_free_struct], ['pf_', 'po_']):
            # iterate over the entries of the struct
            for key in symsstruct.keys():
                _entry = symsstruct[key]

                # check if the entry is a list or a casadi SX
                assert type(_entry) in [list, ca.SX], "The symbolic struct has to be a list[SX] or a casadi SX"
                if isinstance(_entry, list):
                    # unpack the list
                    for index, element in enumerate(_entry):
                        assert type(element) == ca.SX, "The entries of the list have to be casadi SX"
                        return_list.append(prefix + key + f'_{index}')
                else:
                    return_list.append(prefix + key)

        return return_list

    def declareSyms(self, symParams: List[SymbolicParameter]):
        """ Declares some of the attributes as symbolic (optimization) variables that are collected in a struct .syms,
         the variables can still be accessed as attributes of the class. Can only be called once."""

        # assert self.syms is None, 'The symbolic variables have already been used and cannot be changed'
        for param in symParams:
            assert isinstance(param,SymbolicParameter), "The parameters have to be of type SymbolicParameter (or a subclass)"
            assert param.name in self.__dict__.keys(), f"Variable {param.name} is not a parameter of the class"
            assert param.type in [SymbolicParameter.FIX,
                                  SymbolicParameter.FREE], f"Type has to be either {SymbolicParameter.FIX} or {SymbolicParameter.FREE}"

        # divide the list of symbolic parameters into fixed and free parameters
        symParams_fix = [p for p in symParams if p.type == SymbolicParameter.FIX]
        symParams_free = [p for p in symParams if p.type == SymbolicParameter.FREE]

        # collect the entries
        entries_fix = [p.entry for p in symParams_fix]
        entries_free = [p.entry for p in symParams_free]

        # create the struct
        self._syms_fix_struct = struct_symSX(entries_fix)
        self._syms_free_struct = struct_symSX(entries_free)

        # store the symbolic variables in a dictionary
        self._syms_dict_instance_fix = {p.name: p for p in symParams_fix}
        self._syms_dict_instance_free = {p.name: p for p in symParams_free}
        self._refreshSymbolicAttributeReferences()

        self.state = self.StateEnum.SYMS_DECLARED

    def replaceFreeSyms(self, sxvector: ca.SX):
        """ Returns a new instance of the parameters class with the symbolic variables replaced by the symbolic values in the provided vector.
          """
        assert sxvector.shape[
                   0] == self._syms_free_struct.size, f"The provided vector ({sxvector.shape}) has to have the same size as the symbolic struct ({self._syms_free_struct.size})"
        # newInstance = Parameters()
        # # copy all attributes
        # for key in self.__dict__.keys():
        #     setattr(newInstance, key, object.__getattribute__(self, key))
        # newInstance._syms_free_struct = self._syms_free_struct(sxvector)
        # newInstance._refreshSymbolicAttributeReferences()
        # return newInstance

        self._syms_free_struct = self._syms_free_struct(sxvector)
        self.state = self.StateEnum.SYMS_REPLACED
        self._refreshSymbolicAttributeReferences()


    def addSymsOptimalValues(self, optimalValues_freeSyms: Union[ca.DM, ca.tools.struct] = None,
                             optimalValues_fixSyms: Union[ca.DM, ca.tools.struct] = None):
        """
        Adds the optimal values to the class. The optimal values can be provided as a casadi DM or a casadi struct.
        The optimal values are stored in both the class attributes and the symbolic struct syms_opt. This also removes
        the symbolic struct syms from the class, as the optimal values are now stored in syms_opt.

        Usefull for postprocessing the results of an optimization problem.

        :param optimalValues: The optimal values of the symbolic variables, either as a casadi DM or a casadi struct.
        """

        logging.info('Adding optimal values to the parameters class ...')
        if optimalValues_fixSyms is None:
            assert self._syms_fix_struct.size == 0, "The fixed symbolic struct has to be empty if no optimal values are provided"
            optimalValues_fixSyms = self.syms_fix_struct(0)
        if optimalValues_freeSyms is None:
            assert self._syms_free_struct.size == 0, "The free symbolic struct has to be empty if no optimal values are provided"
            optimalValues_freeSyms = self.syms_free_struct(0)

        # FIXED SYMBOLIC PARAMETERS
        if type(optimalValues_fixSyms) == ca.DM:
            assert optimalValues_fixSyms.shape[
                       0] == self.syms_fix_struct.size, "The provided vector has to have the same size as the symbolic struct"
            struct_opt_fix = self.syms_fix_struct(optimalValues_fixSyms)
        elif type(optimalValues_fixSyms) == ca.tools.structure3.DMStruct:
            assert optimalValues_fixSyms.keys() == self.syms_fix_struct.keys(), "The provided struct has to have the same entries as the symbolic struct"
            struct_opt_fix = optimalValues_fixSyms
        else:
            raise TypeError(
                f"The optimal FIXED values provided have to be a casadi DM or a casadi struct, but are of type {type(optimalValues_freeSyms)}")

        # FREE SYMBOLIC PARAMETERS
        if type(optimalValues_freeSyms) == ca.DM:
            assert optimalValues_freeSyms.shape[
                       0] == self.syms_free_struct.size, "The provided vector has to have the same size as the symbolic struct"
            struct_opt_free = self.syms_free_struct(optimalValues_freeSyms)
        elif type(optimalValues_freeSyms) == ca.tools.struct:
            assert optimalValues_freeSyms.keys() == self.syms_free_struct.keys(), "The provided struct has to have the same entries as the symbolic struct"
            struct_opt_free = optimalValues_freeSyms
        else:
            raise TypeError(
                f"The optimal FREE values provided have to be a casadi DM or a casadi struct, but are of type {type(optimalValues_freeSyms)}")

        # overwrite (again) to refresh the reference
        for symstruct in [struct_opt_fix, struct_opt_free]:
            for key in symstruct.keys():
                # cast to correct type and set the attribute
                newValue = self._typesDict[key][0](symstruct[key])
                setattr(self, key, newValue)

                value_formatted = str(newValue).replace("\n", "")
                logging.info(f'\t - Replaced symbolic variable for parameter {key} with value {value_formatted}!')

                # add the name to the list of optimized variables
                self._optimized_syms_names.append(key)
        # self._syms_free_struct = None

        # FIND DANGLING SYMBOLIC EXPRESSIONS
        for key in self.__dict__.keys():

            # ignore private attributes
            if key.startswith('_'):
                continue

            param = self.__dict__[key]
            if type(param) == ca.SX:
                assert ca.which_depends(param, self._syms_free_struct) or ca.which_depends(param, self._syms_fix_struct)

                # create a function that replaces the symbolic variables with the optimal values
                f = ca.Function('f', [self._syms_fix_struct, self._syms_free_struct], [param],
                                ['p_fix_opt', 'p_free_opt'], ['p'])
                newVal = f(struct_opt_fix, struct_opt_free)

                # cast to correct type and set the attribute
                newVal = self._typesDict[key][0](newVal)
                setattr(self, key, newVal)

                logging.info(f'\t - Replaced dangling symbolic expression for parameter {key} with value {newVal}!')

        # remove the symbolic struct from the class (make it empty)
        self._syms_free_struct = struct_symSX([])
        self._syms_fix_struct = struct_symSX([])

        # store the optimal values
        self._syms_free_struct_opt = struct_opt_free
        self._syms_fix_struct_opt = struct_opt_fix

        self.state = self.StateEnum.OPTIMAL_VALUES_ADDED

        logging.info('... Done!')

    def require(self, name):
        """ Check if a parameter of the name has a value, if not raise an error """
        if name not in self.__dict__.keys():
            raise AttributeError(f"Parameter '{name}' is not known to the parameter class!")
        if isinstance(self.__dict__[name], NotDefinedParameter):
            raise AttributeError(f"A value for the parameter '{name}' is required!")

    def getSymsListByNames(self, names: List[str]) -> Tuple[List[ca.SX], List[str], List]:
        """ Obtains a list of symbolic variables by their names, if the variable of the given name is not a symbolic variable, it is not returned.

        Returns a tuple of two lists,
            - the first list contains the symbolic variables,
            - the second list contains the names of the variables.
         """
        return_list = []
        return_list_names = []
        return_list_default_values = []

        for symsstruct in [self._syms_fix_struct, self._syms_free_struct]:
            for name in names:
                if name in symsstruct.keys():
                    return_list.append(symsstruct[name])
                    return_list_names.append('p_' + name)
                    return_list_default_values.append(self._syms_dict_all[name].defaultValue)

        assert len(return_list) == len(
            return_list_names), "The number of symbolic variables and their names has to be equal"
        assert len(return_list) == len(
            return_list_default_values), "The number of symbolic variables and their default values has to be equal"

        return return_list, return_list_names, return_list_default_values

    @property
    def _syms_dict_all(self) -> Dict[str, SymbolicParameter]:
        return {**self._syms_dict_instance_fix, **self._syms_dict_instance_free}

    def _refreshSymbolicAttributeReferences(self):
        # link the symbolic variable to the attributes of the class
        for key in self._syms_free_struct.keys():
            setattr(self, key, self._syms_free_struct[key])

        for key in self._syms_fix_struct.keys():
            setattr(self, key, self._syms_fix_struct[key])

    def __getattribute__(self, name):
        """custom wrapper that forces the user to define the variable values before using them"""
        attribute = object.__getattribute__(self, name)  # avoid recursion
        if isinstance(attribute, NotDefinedParameter):
            raise AttributeError(f"Parameter '{name}' is not defined!")

        return attribute

    def __setattr__(self, key, value):
        """ Custom wrapper to check if the name of the  parameter is known,
        the type of the value is correct and to store the type of the parameter.

        All parameter have initially the type NotDefinedParameter,
        which is a helper class to store the type and description of the parameter.
        """

        # if it is the type dict, we just store
        if key == '_typesDict':
            object.__setattr__(self, key, value)
            return

        # if what we set is a NotDefinedParameter, store the type  and the description
        if isinstance(value, NotDefinedParameter):
            type_dict = object.__getattribute__(self, '_typesDict')
            desc_dict = object.__getattribute__(self, '_descriptionDict')
            type_dict[key] = value.types  # store the type
            desc_dict[key] = value.description  # store the descriptions
            object.__setattr__(self, key, value)  # write the attribute
            return

        # check if the type is correct
        type_dict = object.__getattribute__(self, '_typesDict')
        if key in type_dict.keys():
            assert type(value) in type_dict[
                key], f"Parameter '{key}' has to be of type {type_dict[key]} but is of type {type(value)}"
        object.__setattr__(self, key, value)

    def __str__(self):
        """ Creates a nice string representation of the class."""

        LEN_DESCRIPTION = 60
        LEN_NAME = 10
        returnstr = (f"Parameter class with the "
                     f"following attributes:\n")

        # print all attribute of the class
        for key in self.__dict__.keys():
            if not key.startswith('_'):
                variable_repr = repr(self.__dict__[key]).replace('\n', '')  # remove \n from rep
                variable_descr = self._descriptionDict.get(key, "")
                variable_opt = ' (OPT) ' if key in self._optimized_syms_names else ' '
                variable_name = key
                returnstr += (f"     {variable_descr}{' ' * (LEN_DESCRIPTION - len(variable_descr))}|"
                              f"{' ' * (7 - len(variable_opt))}{variable_opt} {variable_name}{' ' * (LEN_NAME - len(variable_name))}"
                              f" = {variable_repr}\n")
        return returnstr

    def __repr__(self):
        return self.__str__()


class Parameters(ParametersBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = NotDefinedParameter("The pertubation parameter", [float, int, ca.SX, ca.DM])
        self.tauf = NotDefinedParameter("The horizon of the Problem", [float, int, ca.SX, ca.DM])
        self.dControl = NotDefinedParameter("The number of coefficients in the control polynomial", [int, ca.DM])
        self.U_cycles = NotDefinedParameter("The coefficients of the control polynomial", [list])
        self.Nmicro = NotDefinedParameter("The number of integration steps of the micro-integrations", [int])
        self.Nmacro = NotDefinedParameter("The number of integration steps of the macro-integrations", [int])
        self.dmacro = NotDefinedParameter("The number of stages in the macro-integration", [int])
        self.Nctrl = NotDefinedParameter("The number of parameters of the control parametrization", [int])

        # iterate over the kwargs
        for key, value in kwargs.items():
            assert not key.startswith('_'), "The parameter name cannot start with an underscore"
            assert key in self.__dict__.keys(), f"Variable {key} is not a parameter of the class"

            # set the value
            setattr(self, key, value)
