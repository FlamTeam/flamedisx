import flamedisx as fd
import pandas as pd
import random
import string
import tensorflow as tf
from textwrap import dedent
import typing as ty

from boltons.funcutils import FunctionBuilder

export, __all__ = fd.exporter()


@export
def log_likelihood(
            sources: ty.Dict[str, fd.ERSource],
            data: pd.DataFrame,
            minus_two=False,
            source_params=None,
            tensorflow_function=True,
            **common_params) -> ty.Callable:

        if source_params is None:
            source_params = dict()

        fname = ('log_likelihood_'
                 + ''.join(random.choices(string.ascii_lowercase, k=10)))
        ll = FunctionBuilder(fname)
        for sname in sources:
            ll.add_arg(sname + '_rate_multiplier',
                       default=tf.constant(1., dtype=tf.float32))

        # Check defaults for common parameters are consistent between
        # sources
        common_defaults = dict()
        for pname in common_params:
            defs = [s.defaults[pname] for s in sources.values()]
            if len(set([x.numpy() for x in defs])) > 1:
                raise ValueError(
                    f"Inconsistent defaults {defs} for common parameters")
            common_defaults[pname] = defs[0]

        function_body = dedent("""
        mu = tf.constant(0., dtype=tf.float32)
        lls = tf.zeros({n_events}, dtype=tf.float32)
        """).format(n_events=len(data))

        kwarg_template = "{kwargname}={pname}"
        for sname, s in sources.items():
            # Have to copy since data is modified by set_data
            s.set_data(data.copy())

            callstring = []

            source_params.setdefault(sname, dict())
            for pname in source_params[sname]:
                kwargname = sname + '_' + pname
                default = s.defaults[kwargname]
                ll.add_arg(kwargname, default=default)
                callstring.append(kwarg_template.format(
                    kwargname=kwargname, pname=pname))

            for pname in common_params:
                if pname not in ll.defaults:
                    ll.add_arg(pname, default=common_defaults[pname])
                callstring.append(kwarg_template.format(
                    kwargname=pname, pname=pname))

            source_params[sname].update(common_params)

            function_body += dedent("""
            mu += ({fname}.mu_itps['{sname}']({callstring})
                   * {sname}_rate_multiplier)
            lls += {fname}.sources['{sname}'].likelihood({callstring})
            """).format(fname=fname,
                        sname=sname,
                        callstring=', '.join(callstring))

        prefactor = -2. if minus_two else 1.
        function_body += (
            f"\nreturn {prefactor} * (-mu + tf.reduce_sum(fd.tf_log10(lls)))")

        ll.body = function_body
        f = ll.get_func(execdict=dict(tf=tf, fd=fd))
        f.mu_itps = {
            sname: s.mu_interpolator(**source_params[sname])
            for sname, s in sources.items()}
        f.sources = sources

        if tensorflow_function:
            return tf.function(f)
        return f
