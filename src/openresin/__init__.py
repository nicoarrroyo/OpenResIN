"""OpenResIN — Open-source Reservoir Identifier and Navigator.

A four-stage pipeline for identifying small water reservoirs in Sentinel-2
imagery: label, train, predict, evaluate. Each stage is a module with a
``main()`` entry point, registered as a console script in ``pyproject.toml``.

Deliberately empty of imports. Importing submodules here would run them at
package-import time, which is the coupling this package layout exists to
remove.
"""
