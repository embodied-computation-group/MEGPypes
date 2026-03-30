from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nipype import config as nconfig
from nipype import logging as nlogging

from megpypes.pipelines.meg_preprocessing import create_meg_preprocessing


@dataclass(frozen=True)
class PipelineRunResult:
    workflow: Any
    result: Any
    plugin: str
    n_workers: int


class MegPypesRunner:
    """High-level API for running the MEG preprocessing pipeline from code or notebooks."""
    
    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.graph_path: Path | None = None
        self._report_process: subprocess.Popen[str] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MegPypesRunner":
        with Path(config_path).open("r", encoding="utf-8") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return cls(config=config, config_path=config_path)

    @property
    def paths_config(self) -> dict[str, Any]:
        return dict(self.config.get("paths", {}))

    @property
    def workflow_config(self) -> dict[str, Any]:
        return dict(self.config.get("workflow", {}))

    @property
    def pipeline_config(self) -> dict[str, Any]:
        return dict(self.config.get("pipeline_config", {}))

    def configure_nipype_logging(
        self,
        *,
        crashdump_dir: str | Path = "crashes",
        interface_level: str = "DEBUG",
        workflow_level: str = "DEBUG",
        remove_unnecessary_outputs: bool = False,
    ) -> Path:
        paths = self.paths_config
        logs_dir = Path(paths["workdir"]) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "logging": {
                "log_directory": str(logs_dir),
                "log_to_file": True,
                "interface_level": interface_level,
                "workflow_level": workflow_level,
            },
            "execution": {
                "crashdump_dir": os.path.abspath(str(crashdump_dir)),
                "remove_unnecessary_outputs": remove_unnecessary_outputs,
            },
        }

        nconfig.update_config(config_dict)
        nlogging.update_logging(nconfig)
        return logs_dir

    def create_workflow(
        self,
        *,
        paths_override: dict[str, Any] | None = None,
        pipeline_config_override: dict[str, Any] | None = None,
    ) -> Any:
        paths = self.paths_config
        if paths_override:
            paths.update(paths_override)

        pipeline_config = self.pipeline_config
        if pipeline_config_override:
            pipeline_config.update(pipeline_config_override)

        required_keys = [
            "basedir",
            "workdir",
            "outputdir",
            "file_templates",
            "iterable_fields",
            "iterable_values",
        ]
        missing = [key for key in required_keys if key not in paths]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required paths configuration keys: {missing_str}")

        return create_meg_preprocessing(
            basedir=paths["basedir"],
            workdir=paths["workdir"],
            output_dir=paths["outputdir"],
            file_templates=paths["file_templates"],
            iterable_fields=paths["iterable_fields"],
            iterable_values=paths["iterable_values"],
            pipeline_config=pipeline_config,
        )

    def resolve_workers(self, requested_workers: int | None = None) -> int:
        if requested_workers is not None:
            return max(1, int(requested_workers))

        wf_config = self.workflow_config
        cpu_default = max(1, (os.cpu_count() or 1) - 2)

        if wf_config.get("auto_workers", False):
            return cpu_default

        return max(1, int(wf_config.get("n_workers", cpu_default)))

    def write_graph(
        self,
        workflow: Any,
        *,
        graph2use: str = "colored",
        simple_form: bool = True,
    ) -> Path:
        workflow.write_graph(graph2use=graph2use, simple_form=simple_form)
        graph_path = Path(workflow.base_dir) / workflow.name / "graph.png"
        self.graph_path = graph_path
        return graph_path

    def run(
        self,
        *,
        workflow: Any | None = None,
        n_workers: int | None = None,
        plugin: str | None = None,
        write_graph: bool = False,
        plugin_args: dict[str, Any] | None = None,
        paths_override: dict[str, Any] | None = None,
        pipeline_config_override: dict[str, Any] | None = None,
    ) -> PipelineRunResult:
        self.configure_nipype_logging()

        wf = workflow or self.create_workflow(
            paths_override=paths_override,
            pipeline_config_override=pipeline_config_override,
        )

        if write_graph:
            self.write_graph(wf)

        resolved_workers = self.resolve_workers(n_workers)
        selected_plugin = plugin or self.workflow_config.get("plugin", "Linear")
        selected_plugin_args = plugin_args or {"n_procs": resolved_workers}

        result = wf.run(
            plugin=selected_plugin,
            plugin_args=selected_plugin_args,
        )

        return PipelineRunResult(
            workflow=wf,
            result=result,
            plugin=selected_plugin,
            n_workers=resolved_workers,
        )

    def launch_report_app(
        self,
        *,
        report_root: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8501,
        headless: bool = True,
        app_path: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen[str]:
        if self._report_process is not None and self._report_process.poll() is None:
            self.stop_report_app()

        app_file = Path(app_path) if app_path else Path(__file__).parent / "report" / "report_app.py"
        app_file = app_file.resolve()
        if not app_file.exists():
            raise FileNotFoundError(f"Could not find Streamlit app at {app_file}")

        repo_root = Path(__file__).resolve().parents[2]

        env = os.environ.copy()
        if report_root is not None:
            root_candidate = Path(report_root).expanduser()
            if not root_candidate.is_absolute():
                root_candidate = (repo_root / root_candidate).resolve()
            else:
                root_candidate = root_candidate.resolve()
            env["MEGPYPES_REPORT_ROOT"] = str(root_candidate)

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            str(headless).lower(),
            "--browser.gatherUsageStats",
            "false",
        ]
        if extra_args:
            cmd.extend(extra_args)

        self._report_process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self._report_process

    def stop_report_app(self, timeout: float = 3.0) -> None:
        process = self._report_process
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)

    @staticmethod
    def report_url(*, host: str = "127.0.0.1", port: int = 8501) -> str:
        return f"http://{host}:{port}"


__all__ = ["MegPypesRunner", "PipelineRunResult"]
