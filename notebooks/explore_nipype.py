import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # set wd
    import os
    from pathlib import Path
    print(Path.cwd())
    wd = Path.cwd()
    if wd.name == "notebooks":
        os.chdir(wd.parent)
    print(f"Working Dir Base: {(Path.cwd())}")
    return (os,)


@app.cell
def _():
    import yaml
    import time
    from bids.layout import BIDSLayout
    # import
    from megpypes.pipelines.init_preproc import create_initial_preprocessing
    from nipype import config as nconfig

    return BIDSLayout, create_initial_preprocessing, nconfig, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    loaded a MEG dataset from here:
    https://openneuro.org/datasets/ds006629/versions/1.0.1

    using datalad
    ```
    datalad install -s https://github.com/OpenNeuroDatasets/ds006629.git data/ds006629
    ```
    ```
    cd output/ds006629
    datalad get -r .
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we need to grab data from our dataset.
    We assume the data operate within a bids compliant data structure
    https://peerherholz.github.io/workshop_weizmann/nipype/notebooks/basic_data_input_bids.html
    """)
    return


@app.cell
def _(BIDSLayout):
    # do some data inspection using pybids?
    layout = BIDSLayout("data/ds006629/")
    print(layout)
    # subjects
    subjects = layout.get_subjects()
    print(f"Subjects: {subjects}")
    # datatypes
    bidstypes = layout.get_datatypes()
    print(f"Data types: {bidstypes}")
    # suffixes
    print(layout.get_suffixes(datatype='func'))
    # see tasks
    layout.get_tasks()
    # see dataset description
    layout.get_dataset_description()
    #

    # see data metadata
    return


@app.cell
def _(create_initial_preprocessing, nconfig, os, yaml):
    # Load configs
    config_path = 'config/config_ds006629.yaml'
    with open(config_path, 'r') as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    wf_config = config['workflow']
    paths_config = config['paths']
    preproc_args = config["init_preproc"]['args']
    preproc_steps = config["init_preproc"]['steps']


    nconfig.update_config(
        {
            'logging': 
                {'log_directory': os.path.join(paths_config['workdir'], 'logs'), 
                'log_to_file': True, 
                'interface_level': 'info', 
                'workflow_level': 'info'}, 
                'execution': 
                    {'crashdump_dir': os.path.abspath('crashes'), 
                    'remove_unnecessary_outputs': False
                    }
            })

    # Create workflow
    wf = create_initial_preprocessing(
        basedir=paths_config['basedir'], 
        workdir=paths_config['workdir'], 
        output_dir=paths_config['outputdir'], 
        subject_list=paths_config['subjects'], 
        stepflags_params=preproc_steps, 
        crop_params=preproc_args['crop'], 
        filter_params=preproc_args['filter'], 
        gradcomp_params=preproc_args['gradcomp']
        )
    
    # visualize workflow graph
    wf.write_graph(graph2use='colored', simple_form=True)
    print(f'Workflow graph saved to: {wf.base_dir}/megpreproc/graph.png')

    # Configure Nipype logging (applies to all subprocesses)
    n_workers = wf_config.get('n_workers', max(1, os.cpu_count() - 2))
    print(f'Running with {n_workers} workers')

    # Run workflow
    result = wf.run(plugin=wf_config['plugin'], plugin_args={'n_procs': n_workers})  # ← Now actually used!
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
