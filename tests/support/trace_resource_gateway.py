"""Real gateway/recovery algorithms over in-memory spans and local artifacts."""

from mlflow.entities import TraceState

from mlflow_import.mlflow.gateway import MlflowGateway
from tests.support.fake_gateway import FakeMlflowClient, FakeMlflowException, fake_version_transport


class LocalTraceClient(FakeMlflowClient):
    def get_experiment(self, experiment_id):
        self.calls.append(("get_experiment", {"experiment_id": experiment_id}))
        for experiment in self._experiments_by_name.values():
            if experiment.experiment_id == experiment_id:
                return experiment
        raise FakeMlflowException("synthetic experiment missing")


class LocalTraceGateway(MlflowGateway):
    """Exercise the production gateway, with no HTTP or tracing exporter."""

    def __init__(self, client, *, dry_run=False):
        super().__init__(client, trace_api=client.trace_api, dry_run=dry_run, version_transport=fake_version_transport)

    def create_trace(self, request):
        created = super().create_trace(request)
        trace = self._client._traces[created.trace_id]
        trace.info.state = TraceState(trace.info.state)
        # FakeTraceData mirrors the public span surface, so supply its JSON
        # transport representation for real replacement-reference hashing.
        trace.data.to_dict = lambda: {"spans": [vars(span) for span in trace.data.spans]}
        return created
