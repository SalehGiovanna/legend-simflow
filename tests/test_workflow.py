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
                configfiles=(dummyprod / "simflow-config-stp.yaml",)
            ),
            storage_settings=smkapi.StorageSettings(),
            resource_settings=smkapi.ResourceSettings(cores=1),
        )
        dag = wf_api.dag()
        dag.execute_workflow()
