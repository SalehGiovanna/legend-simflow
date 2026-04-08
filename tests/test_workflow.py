from __future__ import annotations

from pathlib import Path

import pytest
from snakemake import api as smkapi

dummyprod = Path(__file__).parent / "dummyprod"


def test_dag():
    output = smkapi.OutputSettings(verbose=False)

    # build workflow and DAG, execute with touch executor (no remage needed)
    with smkapi.SnakemakeApi(output) as api:
        wf_api = api.workflow(
            snakefile=dummyprod / "workflow/Snakefile",
            workdir=dummyprod,
            config_settings=smkapi.ConfigSettings(
                configfiles=(dummyprod / "simflow-config.yaml",)
            ),
            storage_settings=smkapi.StorageSettings(),
            resource_settings=smkapi.ResourceSettings(cores=1),
        )
        dag = wf_api.dag()
        dag.execute_workflow(executor="touch")


@pytest.mark.needs_remage
def test_stp_workflow():
    output = smkapi.OutputSettings(verbose=False)

    with smkapi.SnakemakeApi(output) as api:
        wf_api = api.workflow(
            snakefile=dummyprod / "workflow/Snakefile",
            workdir=dummyprod,
            config_settings=smkapi.ConfigSettings(
                configfiles=(dummyprod / "simflow-config.yaml",),
                config={
                    "experiment": "l200p03",
                    "runlist": ["l200-p03-r000-phy"],
                    "make_steps": ["vtx", "stp"],
                    "benchmark": {"enabled": True, "n_primaries": {"stp": 1000}},
                },
            ),
            storage_settings=smkapi.StorageSettings(),
            resource_settings=smkapi.ResourceSettings(cores=1),
        )
        dag = wf_api.dag()
        dag.execute_workflow()
